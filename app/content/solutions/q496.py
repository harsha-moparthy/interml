import numpy as np


def q496(series, alpha):
    """Return recursively smoothed observations, initializing with the first value and
    using alpha*x[t]+(1-alpha)*s[t-1]. Require 0<alpha<=1.
    """
    if not 0 < alpha <= 1:
        raise ValueError("invalid alpha")
    x = np.asarray(series, float)
    out = x.copy()
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out
