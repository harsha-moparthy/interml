import torch
from torch.nn import functional as F


def q528(logits, mask):
    """SPLADE-style token pooling for [B,L,V] logits and Boolean [B,L] mask. Return
    max-pooled log1p(relu) weights, mean document L1 mass, and FLOPS surrogate
    sum_v(mean_b weights)^2. All-padding documents become zero.
    """
    activated = torch.log1p(F.relu(logits)).masked_fill(~mask[:, :, None], 0)
    weights = activated.max(dim=1).values
    return (weights, weights.sum(-1).mean(), weights.mean(0).square().sum())
