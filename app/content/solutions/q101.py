import numpy as np


def q101(function, x, epsilon=1e-05):
    """Estimate the gradient of a scalar-valued function at a floating-point array using
    central finite differences. Do not mutate the caller's array.
    """
    x = np.array(x, float, copy=True)
    gradient = np.zeros_like(x)
    for index in np.ndindex(x.shape):
        plus = x.copy()
        minus = x.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        gradient[index] = (function(plus) - function(minus)) / (2 * epsilon)
    return gradient
