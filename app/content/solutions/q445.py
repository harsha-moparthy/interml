"""Implement causal FlashAttention forward for one contiguous CUDA float16 head Q,K,V(N,D),
with D in {16,32,64,128}. Use query/key tiles of 32, float32 online-softmax state, and
no stored score matrix. Return a float16 output.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def flash_kernel(Q, K, V, O, N: tl.constexpr, D: tl.constexpr):
    rows = tl.program_id(0) * 32 + tl.arange(0, 32)
    cols = tl.arange(0, 32)
    ds = tl.arange(0, D)
    q = tl.load(Q + rows[:, None] * D + ds[None, :], rows[:, None] < N, 0)
    m = tl.full((32,), -float("inf"), tl.float32)
    l = tl.zeros((32,), tl.float32)
    acc = tl.zeros((32, D), tl.float32)
    for start in range(tl.program_id(0) + 1):
        keys = start * 32 + cols
        k = tl.load(K + keys[None, :] * D + ds[:, None], keys[None, :] < N, 0)
        score = tl.dot(q, k) * (D**-0.5)
        valid = (keys[None, :] < N) & (keys[None, :] <= rows[:, None])
        score = tl.where(valid, score, -float("inf"))
        next_m = tl.maximum(m, tl.max(score, 1))
        p = tl.exp(score - next_m[:, None])
        correction = tl.exp(m - next_m)
        v = tl.load(V + keys[:, None] * D + ds[None, :], keys[:, None] < N, 0)
        acc = acc * correction[:, None] + tl.dot(p.to(tl.float16), v)
        l = l * correction + tl.sum(p, 1)
        m = next_m
    tl.store(
        O + rows[:, None] * D + ds[None, :],
        acc / l[:, None],
        rows[:, None] < N,
    )


def solve(q, k, v):
    assert q.is_cuda and q.shape == k.shape == v.shape
    assert all(
        x.is_contiguous() and x.dtype == torch.float16 and x.device == q.device
        for x in (q, k, v)
    )
    n, d = q.shape
    assert n > 0 and d in (16, 32, 64, 128)
    out = torch.empty_like(q)
    flash_kernel[(triton.cdiv(n, 32),)](q, k, v, out, n, d)
    return out
