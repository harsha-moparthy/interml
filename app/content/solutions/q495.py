import numpy as np


def q495(series, window):
    """Return simple moving averages for complete windows only. Require
    1<=window<=len(series).
    """
    x = np.asarray(series, float)
    if not 1 <= window <= len(x):
        raise ValueError("invalid window")
    sums = np.r_[0, np.cumsum(x)]
    return (sums[window:] - sums[:-window]) / window
