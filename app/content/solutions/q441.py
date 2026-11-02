"""Compute float32 sums of rows of a contiguous CUDA matrix with 1<=columns<=65536, using
one Triton program per row.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(X, Y, C: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, B)
    values = tl.load(X + row * C + j, j < C, 0).to(tl.float32)
    tl.store(Y + row, tl.sum(values, 0))


def solve(x):
    assert x.is_cuda and x.is_contiguous() and x.ndim == 2
    r, c = x.shape
    assert 0 < c <= 65536
    out = torch.empty(r, device=x.device, dtype=torch.float32)
    if r:
        sum_kernel[(r,)](x, out, c, triton.next_power_of_2(c))
    return out
