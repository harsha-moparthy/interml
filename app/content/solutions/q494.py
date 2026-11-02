import numpy as np


def q494(series, lag=1):
    """Return lag-differenced observations x[t]-x[t-lag], omitting unavailable leading
    values. Lag must be positive.
    """
    if lag <= 0:
        raise ValueError("lag must be positive")
    x = np.asarray(series, float)
    return x[lag:] - x[:-lag]
