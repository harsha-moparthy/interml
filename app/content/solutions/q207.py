import numpy as np


def q207(labels, scores):
    """Build a precision-recall curve at each distinct score threshold for binary labels
    with at least one positive. Return recall, precision, and stepwise average
    precision; group tied scores together.
    """
    y, s = (np.asarray(labels, int), np.asarray(scores, float))
    order = np.argsort(-s, kind="stable")
    y, s = (y[order], s[order])
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(y)[ends]
    recall = np.r_[0, tp / y.sum()]
    precision = np.r_[1, tp / (ends + 1)]
    return (
        recall,
        precision,
        float(np.sum(np.diff(recall) * precision[1:])),
    )
