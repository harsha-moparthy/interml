import torch
import torch.nn.functional as F


def contrastive_loss(first, second, temperature=0.1, *, mode="clip"):
    """Paired (N,D) embeddings: symmetric CLIP loss or SimCLR NT-Xent.

    CLIP uses only cross-view negatives. SimCLR also uses within-view negatives,
    excluding self-similarity. Cost/storage: O(N^2*D)/O(N^2).
    """
    if (
        first.ndim != 2
        or first.shape != second.shape
        or len(first) < 2
        or temperature <= 0
    ):
        raise ValueError(
            "matching (N,D) arrays, N>=2, and positive temperature required"
        )
    first, second = F.normalize(first, dim=-1), F.normalize(second, dim=-1)
    count = len(first)
    if mode == "clip":
        scores = first @ second.T / temperature
        labels = torch.arange(count, device=scores.device)
        return (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2
    if mode != "simclr":
        raise ValueError("mode must be clip or simclr")
    features = torch.cat((first, second))
    scores = features @ features.T / temperature
    scores = scores.masked_fill(
        torch.eye(2 * count, device=scores.device, dtype=torch.bool), -torch.inf
    )
    labels = (torch.arange(2 * count, device=scores.device) + count) % (2 * count)
    return F.cross_entropy(scores, labels)
