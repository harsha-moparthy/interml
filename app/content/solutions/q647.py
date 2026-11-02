import numpy as np


def binary_ranking_metrics(labels, scores):
    """Return ROC AUC, trapezoidal PR AUC, and non-interpolated average precision.

    Group score ties before accumulating confusion counts. Both classes required.
    Sorting dominates O(N log N); PR area and AP are deliberately named separately.
    """
    y, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if (
        y.ndim != 1
        or scores.shape != y.shape
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(scores).all()
        or len(np.unique(y)) != 2
    ):
        raise ValueError("finite aligned scores and both binary classes required")
    order = np.argsort(-scores, kind="stable")
    y, scores = y[order], scores[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    true_positive = np.cumsum(y)[ends]
    false_positive = ends + 1 - true_positive
    recall = np.r_[0.0, true_positive / y.sum()]
    false_rate = np.r_[0.0, false_positive / (len(y) - y.sum())]
    precision = np.r_[1.0, true_positive / (ends + 1)]
    roc_area = np.sum(np.diff(false_rate) * (recall[1:] + recall[:-1]) / 2)
    pr_area = np.sum(np.diff(recall) * (precision[1:] + precision[:-1]) / 2)
    average_precision = np.sum(np.diff(recall) * precision[1:])
    return float(roc_area), float(pr_area), float(average_precision)
