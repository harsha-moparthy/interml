import torch


class BatchNorm(torch.nn.Module):
    """BatchNorm for (N,C), (N,C,L), or (N,C,H,W); tracks EMA running statistics.

    Training normalizes with biased variance but stores the unbiased estimate,
    matching PyTorch. Evaluation uses buffers. Half precision accumulates in fp32.
    """

    def __init__(self, channels, eps=1e-5, momentum=0.1):
        super().__init__()
        if channels < 1 or eps <= 0 or not 0 <= momentum <= 1:
            raise ValueError("invalid normalization settings")
        self.eps, self.momentum = eps, momentum
        self.weight = torch.nn.Parameter(torch.ones(channels))
        self.bias = torch.nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def forward(self, x):
        if x.ndim not in (2, 3, 4) or x.shape[1] != len(self.weight):
            raise ValueError("expected (N,C), (N,C,L), or (N,C,H,W)")
        work = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        axes = (0,) + tuple(range(2, x.ndim))
        shape = (1, -1) + (1,) * (x.ndim - 2)
        if self.training:
            count = x.numel() // x.shape[1]
            if count <= 1:
                raise ValueError("training needs more than one value per channel")
            mean = work.mean(axes)
            variance = work.var(axes, unbiased=False)
            with torch.no_grad():
                self.running_mean.lerp_(
                    mean.detach().to(self.running_mean.dtype), self.momentum
                )
                self.running_var.lerp_(
                    (variance.detach() * count / (count - 1)).to(
                        self.running_var.dtype
                    ),
                    self.momentum,
                )
        else:
            mean, variance = self.running_mean, self.running_var
        normalized = (work - mean.reshape(shape)) * torch.rsqrt(
            variance.reshape(shape) + self.eps
        )
        return (normalized * self.weight.reshape(shape) + self.bias.reshape(shape)).to(
            x.dtype
        )
