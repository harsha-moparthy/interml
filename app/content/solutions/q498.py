import numpy as np


def q498(series, lookback, horizon, stride=1):
    """Create chronological input windows and immediately following multi-step targets from
    a time-major array. Return correctly shaped empty arrays when no full window fits.
    """
    x = np.asarray(series)
    if min(lookback, horizon, stride) < 1:
        raise ValueError("positive window sizes required")
    starts = list(range(0, len(x) - lookback - horizon + 1, stride))
    inputs = np.empty((len(starts), lookback) + x.shape[1:], dtype=x.dtype)
    targets = np.empty((len(starts), horizon) + x.shape[1:], dtype=x.dtype)
    for i, s in enumerate(starts):
        inputs[i] = x[s : s + lookback]
        targets[i] = x[s + lookback : s + lookback + horizon]
    return (inputs, targets)
