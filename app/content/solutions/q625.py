import torch


def label_smoothed_cross_entropy(logits, targets, smoothing=0.1):
    """Mean loss for logits (N,C) and integer labels (N,).

    Mix the one-hot target with uniform mass across ALL C classes, matching PyTorch.
    """
    if logits.ndim != 2 or targets.shape != (len(logits),) or not 0 <= smoothing <= 1:
        raise ValueError("invalid shapes or smoothing")
    log_probs = logits.log_softmax(-1)
    hard_loss = -log_probs.gather(1, targets[:, None]).squeeze(1)
    uniform_loss = -log_probs.mean(-1)
    return ((1 - smoothing) * hard_loss + smoothing * uniform_loss).mean()
