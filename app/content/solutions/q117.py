import numpy as np


def q117(train, test):
    """Fit per-column mean imputation on training data and apply it to train/test matrices.
    Use zero for a column that is entirely missing in training.
    """
    a, b = (np.asarray(train, float), np.asarray(test, float))
    count = np.isfinite(a).sum(0)
    means = np.divide(
        np.nansum(a, axis=0),
        count,
        out=np.zeros(a.shape[1]),
        where=count > 0,
    )
    return (
        np.where(np.isnan(a), means, a),
        np.where(np.isnan(b), means, b),
    )
