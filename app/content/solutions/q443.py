"""Multiply contiguous CUDA float16 matrices A(M,K) and B(K,N) using 32x32x32 Triton tiles
and float32 accumulators. Handle nonmultiple boundary sizes; dimensions are positive.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def mm_kernel(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    r = tl.program_id(0) * 32 + tl.arange(0, 32)
    c = tl.program_id(1) * 32 + tl.arange(0, 32)
    z = tl.arange(0, 32)
    acc = tl.zeros((32, 32), tl.float32)
    for start in range(tl.cdiv(K, 32)):
        kk = start * 32 + z
        a = tl.load(
            A + r[:, None] * K + kk[None, :],
            (r[:, None] < M) & (kk[None, :] < K),
            0,
        )
        b = tl.load(
            B + kk[:, None] * N + c[None, :],
            (kk[:, None] < K) & (c[None, :] < N),
            0,
        )
        acc += tl.dot(a, b)
    tl.store(
        C + r[:, None] * N + c[None, :],
        acc,
        (r[:, None] < M) & (c[None, :] < N),
    )


def solve(a, b):
    assert (
        a.is_cuda and b.device == a.device and a.is_contiguous() and b.is_contiguous()
    )
    assert a.dtype == b.dtype == torch.float16
    m, k = a.shape
    kb, n = b.shape
    assert k == kb and min(m, n, k) > 0
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)
    mm_kernel[(triton.cdiv(m, 32), triton.cdiv(n, 32))](a, b, out, m, n, k)
    return out
