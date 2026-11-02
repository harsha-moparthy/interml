import numpy as np


def q198(truth, prediction):
    """Compute macro F1 over the union of observed and predicted labels. Define a class's
    F1 as zero when its denominator is zero.
    """
    y, p = (np.asarray(truth), np.asarray(prediction))
    scores = []
    for label in np.union1d(y, p):
        tp = np.sum((y == label) & (p == label))
        denominator = np.sum(y == label) + np.sum(p == label)
        scores.append(2 * tp / denominator if denominator else 0)
    return float(np.mean(scores)) if scores else 0.0
