import torch
from torch.nn import functional as F


def q507(train, labels, queries, k=5, temperature=0.1):
    """Weighted cosine k-NN classification for disjoint train/query features. Labels are
    contiguous nonnegative class IDs. Clamp k to gallery size; weights are
    exp((similarity-row_max)/temperature). Stable sorting breaks neighbor ties by
    gallery order and class ties by class ID. Return class scores and predictions.
    """
    if not len(train) or k < 1 or temperature <= 0:
        raise ValueError("invalid neighbors/temperature")
    scores = F.normalize(queries, dim=-1) @ F.normalize(train, dim=-1).T
    indices = torch.argsort(scores, dim=1, descending=True, stable=True)[
        :, : min(k, len(train))
    ]
    values = scores.gather(1, indices)
    weights = ((values - values.max(1, keepdim=True).values) / temperature).exp()
    votes = scores.new_zeros((len(queries), int(labels.max()) + 1))
    votes.scatter_add_(1, labels[indices], weights)
    return (votes, votes.argmax(1))
