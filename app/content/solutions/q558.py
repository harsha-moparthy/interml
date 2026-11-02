import torch
from torch.nn import functional as F


def q558(features, bases, k=2, temperature=1.0):
    """Route [B,D] features to supplied orthonormal adapter bases [E,D,r] using squared
    projection norm. Select stable top-k and softmax selected scores; return indices and
    weights. This explicit educational subspace router does not reconstruct an
    unspecified project router.
    """
    if not 1 <= k <= len(bases) or temperature <= 0:
        raise ValueError("invalid k/temperature")
    scores = torch.einsum("bd,edr->ber", features, bases).square().sum(-1)
    ids = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :k]
    return (ids, F.softmax(scores.gather(1, ids) / temperature, -1))
