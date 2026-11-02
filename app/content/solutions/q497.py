import numpy as np


def q497(history, horizon, season_length=1):
    """Return naive and seasonal-naive forecasts. Naive repeats the final value;
    seasonal-naive cycles through the last complete season, supporting multivariate
    time-major arrays.
    """
    x = np.asarray(history)
    if not 1 <= season_length <= len(x) or horizon < 0:
        raise ValueError("invalid horizon or season")
    return (
        np.repeat(x[-1:], horizon, axis=0),
        x[-season_length:][np.arange(horizon) % season_length],
    )
