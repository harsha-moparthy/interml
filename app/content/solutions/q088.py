import numpy as np


def q088(probabilities, uniforms):
    """Sample categorical indices by inverse-CDF sampling from supplied uniforms in [0,1).
    Probabilities are nonnegative and sum to one.
    """
    p = np.asarray(probabilities, float)
    u = np.asarray(uniforms, float)
    if np.any(p < 0) or not np.isclose(p.sum(), 1) or np.any((u < 0) | (u >= 1)):
        raise ValueError("invalid probabilities or uniforms")
    cdf = np.cumsum(p)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, u, side="right")
