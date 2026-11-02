"""Implement inverted dropout for a contiguous CUDA floating tensor using counter-based
seeded random numbers. Do not store a mask; regenerating with the same seed and offsets
must reproduce it.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def dropout_kernel(
    X,
    Y,
    N: tl.constexpr,
    P: tl.constexpr,
    SEED: tl.constexpr,
    B: tl.constexpr,
):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, 0).to(tl.float32)
    keep = tl.rand(SEED, i) > P
    tl.store(Y + i, tl.where(keep, x / (1 - P), 0), i < N)


def solve(x, p=0.5, seed=0):
    assert x.is_cuda and x.is_contiguous() and 0 <= p < 1
    out = torch.empty_like(x)
    if x.numel():
        dropout_kernel[(triton.cdiv(x.numel(), 256),)](x, out, x.numel(), p, seed, 256)
    return out
