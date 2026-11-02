import torch


def q054(x):
    """Implement a custom autograd square operator. Cache the input for backward and return
    2*x times the upstream gradient.
    """

    class Square(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * x

        @staticmethod
        def backward(ctx, gradient):
            (x,) = ctx.saved_tensors
            return 2 * x * gradient

    return Square.apply(x)
