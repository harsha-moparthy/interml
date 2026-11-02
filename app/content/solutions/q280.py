import numpy as np


def q280(prediction, target, epsilon=1e-06):
    """Compute soft Dice loss over all entries using nonnegative predictions and binary
    targets. Add epsilon to numerator and denominator so two empty masks have zero loss.
    """
    p, t = (np.asarray(prediction, float), np.asarray(target, float))
    return float(1 - (2 * np.sum(p * t) + epsilon) / (p.sum() + t.sum() + epsilon))
