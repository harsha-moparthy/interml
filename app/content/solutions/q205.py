import numpy as np


def q205(labels, scores):
    """Construct an ROC curve for binary labels containing both classes. Group equal scores
    into one threshold, prepend the all-negative point, and return FPR, TPR, and
    thresholds.
    """
    y, s = (np.asarray(labels, int), np.asarray(scores, float))
    # Equal scores belong to one threshold; processing ties separately distorts the curve.
    order = np.argsort(-s, kind="stable")
    y, s = (y[order], s[order])
    ends = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp = np.cumsum(y)[ends]
    fp = ends + 1 - tp
    return (
        np.r_[0, fp / (len(y) - y.sum())],
        np.r_[0, tp / y.sum()],
        np.r_[np.inf, s[ends]],
    )
