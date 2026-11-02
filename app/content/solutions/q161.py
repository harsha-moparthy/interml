import numpy as np


def q161(x, y):
    """Find the best regression split over all feature midpoint thresholds by minimizing
    child sum of squared errors. Return feature, threshold, and SSE; return None when no
    split exists.
    """
    x, y = (np.asarray(x, float), np.asarray(y, float))
    best = None
    loss = np.inf
    for j in range(x.shape[1]):
        values = np.unique(x[:, j])
        for threshold in (values[:-1] + values[1:]) / 2:
            left = x[:, j] <= threshold
            score = ((y[left] - y[left].mean()) ** 2).sum() + (
                (y[~left] - y[~left].mean()) ** 2
            ).sum()
            if score < loss:
                best = (j, float(threshold), float(score))
                loss = score
    return best
