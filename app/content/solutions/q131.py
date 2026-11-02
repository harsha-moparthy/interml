import numpy as np


def q131(x, y, threshold=0.999):
    """Flag numeric feature columns whose absolute Pearson correlation with a nonconstant
    target exceeds a threshold. This is a leakage heuristic, not a proof of leakage;
    constant columns get correlation zero.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc = x - x.mean(0)
    yc = y - y.mean()
    denominator = np.linalg.norm(xc, axis=0) * np.linalg.norm(yc)
    correlation = np.divide(
        xc.T @ yc,
        denominator,
        out=np.zeros(x.shape[1]),
        where=denominator > 0,
    )
    return (np.flatnonzero(np.abs(correlation) >= threshold), correlation)
