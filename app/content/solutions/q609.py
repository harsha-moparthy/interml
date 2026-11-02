import torch


class RMSNorm(torch.nn.Module):
    """Normalize the final dimension by RMS without subtracting its mean.

    zero_centered_weight=False uses a Llama-style multiplicative weight initialized
    to one; True uses a Gemma-style (1 + weight) with weight initialized to zero.
    Time/storage: O(number of input elements). Accumulate low precision in fp32.
    """

    def __init__(self, width, eps=1e-6, *, zero_centered_weight=False):
        super().__init__()
        if width <= 0 or eps <= 0:
            raise ValueError("width and eps must be positive")
        self.eps = eps
        self.zero_centered_weight = zero_centered_weight
        self.weight = torch.nn.Parameter(
            torch.zeros(width) if zero_centered_weight else torch.ones(width)
        )

    def forward(self, x):
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        normalized = work * torch.rsqrt(
            work.square().mean(dim=-1, keepdim=True) + self.eps
        )
        scale = 1 + self.weight if self.zero_centered_weight else self.weight
        return (normalized * scale).to(x.dtype)
