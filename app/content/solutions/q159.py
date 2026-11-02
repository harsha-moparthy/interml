import numpy as np


def q159(labels, left_mask):
    """Compute entropy information gain of a Boolean split. Empty children produce zero
    gain; use logarithms base two.
    """

    def entropy(y):
        if len(y) == 0:
            return 0.0
        p = np.unique(y, return_counts=True)[1] / len(y)
        return float(-(p * np.log2(p)).sum())

    y, mask = (np.asarray(labels), np.asarray(left_mask, bool))
    if len(y) == 0 or mask.all() or (not mask.any()):
        return 0.0
    return (
        entropy(y) - mask.mean() * entropy(y[mask]) - (~mask).mean() * entropy(y[~mask])
    )
