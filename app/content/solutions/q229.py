import numpy as np


def q229(logits, labels, alpha=0.25, gamma=2):
    """Compute binary focal loss from logits and 0/1 labels. Weight positive examples by
    alpha and negative examples by 1-alpha.
    """
    x, y = (np.asarray(logits, float), np.asarray(labels, float))
    signed = (2 * y - 1) * x
    logpt = -np.logaddexp(0, -signed)
    weight = np.where(y == 1, alpha, 1 - alpha)
    return float(np.mean(-weight * (-np.expm1(logpt)) ** gamma * logpt))
