import torch


def top_k_router(logits, k=2):
    """Route logits (tokens, experts); return expert IDs, mixture weights, aux loss.

    Weights renormalize over selected experts. The balancing loss uses full
    softmax probabilities so unselected experts can still receive gradients.
    Dispatch/capacity limits belong to the expert layer, not this router.
    """
    if logits.ndim != 2 or not 1 <= k <= logits.shape[1] or len(logits) == 0:
        raise ValueError("expected nonempty (tokens, experts) logits and valid k")
    probabilities = logits.softmax(-1)
    weights, indices = probabilities.topk(k, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    counts = torch.bincount(indices.flatten(), minlength=logits.shape[1])
    fractions = counts.to(probabilities.dtype) / indices.numel()
    loss = logits.shape[1] * (fractions.detach() * probabilities.mean(0)).sum()
    return indices, weights, loss
