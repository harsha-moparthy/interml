import torch


def q532(scores, positives, temperature=0.1):
    """Return multi-positive contrastive loss: negative log of summed positive probability
    per query. Scores/Boolean positives are [B,N], every row needs a positive, and all N
    candidates enter the denominator. Use logsumexp and a positive temperature.
    """
    if (
        temperature <= 0
        or scores.shape != positives.shape
        or (not positives.any(-1).all())
    ):
        raise ValueError("positive labels and temperature required")
    scaled = scores / temperature
    return (
        torch.logsumexp(scaled, -1)
        - torch.logsumexp(scaled.masked_fill(~positives, -torch.inf), -1)
    ).mean()
