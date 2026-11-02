import torch
import torch.nn.functional as F


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Binary focal loss from logits; alpha weights positives, 1-alpha negatives.

    With gamma=0 this becomes alpha-weighted BCE, not unweighted BCE.
    """
    if logits.shape != targets.shape or gamma < 0 or not 0 <= alpha <= 1:
        raise ValueError("invalid focal-loss inputs")
    if not torch.all((targets == 0) | (targets == 1)):
        raise ValueError("focal labels must be binary")
    signed_logits = torch.where(targets.bool(), logits, -logits)
    log_pt = -F.softplus(-signed_logits)
    alpha_t = torch.where(targets.bool(), alpha, 1 - alpha)
    return (-alpha_t * (-torch.expm1(log_pt)).pow(gamma) * log_pt).mean()
