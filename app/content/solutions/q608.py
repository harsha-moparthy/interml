import torch


class LoRALinear(torch.nn.Module):
    """Frozen linear layer plus a trainable low-rank update scaled by alpha/rank.

    merge_weights() is a terminal inference operation: it returns an independent
    frozen Linear, so repeated calls never double-add the update. Continue
    training this wrapper, not the merged copy. Adapter cost is O(rank*(in+out)).
    """

    def __init__(self, base, rank=8, alpha=16.0, std=0.02):
        super().__init__()
        if not isinstance(base, torch.nn.Linear) or rank <= 0 or std <= 0:
            raise ValueError("expected a Linear, positive rank, and positive std")
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.a = torch.nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.b = torch.nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        torch.nn.init.normal_(self.a, std=std)

    def forward(self, x):
        # B starts at zero, so initially this exactly matches the frozen base.
        return self.base(x) + (x @ self.a.T @ self.b.T) * self.scale

    @torch.no_grad()
    def merge_weights(self):
        merged = torch.nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        merged.weight.copy_(self.base.weight + self.scale * (self.b @ self.a))
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged.requires_grad_(False).eval()
