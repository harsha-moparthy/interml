import torch
from torch.nn import functional as F


def q527(features, weight, bias, k=None, threshold=0.0):
    """Apply a nonnegative sparse encoder relu(X@W.T+b). Keep values strictly above
    threshold, then optionally retain top-k coordinates per row with stable index
    tie-breaking. Return dense sparse codes and positive-nonzero counts. Hard masks are
    an inference/training-core choice, not a smooth sparsity penalty.
    """
    if threshold < 0 or (k is not None and k < 0):
        raise ValueError("negative sparsity setting")
    z = F.relu(F.linear(features, weight, bias))
    z = torch.where(z > threshold, z, 0)
    if k is not None:
        indices = torch.argsort(z, dim=-1, descending=True, stable=True)[
            :, : min(k, z.shape[1])
        ]
        mask = torch.zeros_like(z, dtype=torch.bool).scatter_(1, indices, True)
        z = z * mask
    return (z, (z > 0).sum(-1))
