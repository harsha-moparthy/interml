"""Add same-shaped contiguous CUDA Torch tensors with a masked Triton vector kernel; return
an allocated output tensor.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(X, Y, Z, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    tl.store(Z + i, tl.load(X + i, i < N, 0) + tl.load(Y + i, i < N, 0), i < N)


def solve(x, y):
    assert x.is_cuda and x.is_contiguous() and y.is_contiguous()
    assert x.shape == y.shape and x.device == y.device
    out = torch.empty_like(x)
    if x.numel():
        add_kernel[(triton.cdiv(x.numel(), 256),)](x, y, out, x.numel(), 256)
    return out
