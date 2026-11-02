import torch


def batch_hard_triplet_loss(embeddings, labels, margin=0.3):
    """Mine farthest same-class and closest different-class examples for each anchor.

    Use Euclidean distances, exclude the anchor itself, and skip anchors without
    both a positive and a negative. A fully invalid batch has differentiable zero loss.
    """
    if embeddings.ndim != 2 or labels.shape != (len(embeddings),) or margin < 0:
        raise ValueError("invalid embeddings, labels, or margin")
    distances = torch.cdist(embeddings, embeddings)
    same = labels[:, None] == labels[None, :]
    positives = same & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    negatives = ~same
    valid = positives.any(-1) & negatives.any(-1)
    if not valid.any():
        return embeddings.sum() * 0
    hardest_positive = (
        distances.masked_fill(~positives, -torch.inf).max(-1).values[valid]
    )
    hardest_negative = (
        distances.masked_fill(~negatives, torch.inf).min(-1).values[valid]
    )
    return (hardest_positive - hardest_negative + margin).clamp_min(0).mean()
