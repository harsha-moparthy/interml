import numpy as np


def q493(series, lags):
    """Build lag features. Column j contains series[t-lags[j]], and unavailable past values
    are NaN. All lags must be positive.
    """
    x = np.asarray(series, float)
    if any((l <= 0 for l in lags)):
        raise ValueError("lags must be positive")
    out = np.full((len(x), len(lags)), np.nan)
    for j, lag in enumerate(lags):
        if lag < len(x):
            out[lag:, j] = x[:-lag]
    return out
