import numpy as np
from scipy.special import erf


def q220(x):
    """Implement exact GELU, x times the standard normal CDF, using erf."""
    x = np.asarray(x, float)
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))
