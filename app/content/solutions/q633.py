import torch


class SiLU(torch.autograd.Function):
    """Save x only; recompute sigmoid in backward instead of retaining it."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, upstream):
        (x,) = ctx.saved_tensors
        sigmoid = torch.sigmoid(x)
        return upstream * sigmoid * (1 + x * (1 - sigmoid))


def silu(x):
    return SiLU.apply(x)
