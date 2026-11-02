import numpy as np


def q199(truth, prediction):
    """Compute micro F1 for Boolean multilabel indicator matrices by pooling all label
    decisions.
    """
    y, p = (np.asarray(truth, bool), np.asarray(prediction, bool))
    tp = np.sum(y & p)
    denominator = y.sum() + p.sum()
    return float(2 * tp / denominator) if denominator else 0.0
