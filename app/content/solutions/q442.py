"""Compute stable softmax over each row of a finite contiguous CUDA matrix. Support
1<=columns<=65536 and accumulate in float32.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(X, Y, C: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    j = tl.arange(0, B)
    x = tl.load(X + row * C + j, j < C, -float("inf")).to(tl.float32)
    p = tl.exp(x - tl.max(x, 0))
    p = p / tl.sum(p, 0)
    tl.store(Y + row * C + j, p, j < C)


def solve(x):
    assert x.is_cuda and x.is_contiguous() and x.ndim == 2
    r, c = x.shape
    assert 0 < c <= 65536
    out = torch.empty_like(x)
    if r:
        softmax_kernel[(r,)](x, out, c, triton.next_power_of_2(c))
    return out
