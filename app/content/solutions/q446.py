"""Compute LayerNorm backward for contiguous CUDA matrices X and upstream DY plus weight
vector W. Use population variance and epsilon. Return dX, dW, dB; a Triton row kernel
writes partial weight/bias gradients, reduced by Torch.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def ln_backward(
    X,
    DY,
    W,
    DX,
    DW,
    DB,
    C: tl.constexpr,
    EPS: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    j = tl.arange(0, B)
    mask = j < C
    x = tl.load(X + row * C + j, mask, 0).to(tl.float32)
    mean = tl.sum(x, 0) / C
    centered = tl.where(mask, x - mean, 0)
    inverse = tl.rsqrt(tl.sum(centered * centered, 0) / C + EPS)
    norm = centered * inverse
    dy = tl.load(DY + row * C + j, mask, 0).to(tl.float32)
    w = tl.load(W + j, mask, 0).to(tl.float32)
    g = dy * w
    dx = inverse * (g - tl.sum(g, 0) / C - norm * tl.sum(g * norm, 0) / C)
    tl.store(DX + row * C + j, dx, mask)
    tl.store(DW + row * C + j, dy * norm, mask)
    tl.store(DB + row * C + j, dy, mask)


def solve(x, dy, w, eps=1e-5):
    assert x.is_cuda and x.shape == dy.shape and x.ndim == 2
    assert all(t.is_contiguous() and t.device == x.device for t in (x, dy, w))
    r, c = x.shape
    assert r > 0 and 0 < c <= 65536 and w.shape == (c,)
    dx = torch.empty_like(x)
    dw = torch.empty_like(x, dtype=torch.float32)
    db = torch.empty_like(dw)
    ln_backward[(r,)](x, dy, w, dx, dw, db, c, eps, triton.next_power_of_2(c))
    return dx, dw.sum(0).to(w.dtype), db.sum(0).to(w.dtype)
