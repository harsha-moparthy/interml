import numpy as np


def q103(function, x, epsilon=0.0001):
    """Compute a symmetric numerical Hessian of a scalar-valued function at a vector using
    second-order central differences.
    """
    x = np.asarray(x, float)
    n = len(x)
    h = np.zeros((n, n))
    f0 = function(x)
    for i in range(n):
        ei = np.eye(n)[i] * epsilon
        h[i, i] = (function(x + ei) - 2 * f0 + function(x - ei)) / epsilon**2
        for j in range(i):
            ej = np.eye(n)[j] * epsilon
            h[i, j] = h[j, i] = (
                function(x + ei + ej)
                - function(x + ei - ej)
                - function(x - ei + ej)
                + function(x - ei - ej)
            ) / (4 * epsilon**2)
    return h
