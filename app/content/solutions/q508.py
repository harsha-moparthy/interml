from torch.nn import functional as F


def q508(pred1, pred2, target1, target2):
    """Compute symmetric BYOL loss from two online predictions and two target projections
    of equal [B,D] shape. L2-normalize and average cross-view squared distances; stop
    gradients through targets. No contrastive negatives or DINO cross-entropy are used.
    """
    p1, p2 = (F.normalize(pred1, dim=-1), F.normalize(pred2, dim=-1))
    t1, t2 = (
        F.normalize(target1.detach(), dim=-1),
        F.normalize(target2.detach(), dim=-1),
    )
    return ((p1 - t2).square().sum(-1) + (p2 - t1).square().sum(-1)).mean() / 2
