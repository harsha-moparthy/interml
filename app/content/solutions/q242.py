import numpy as np
from scipy.special import expit


def q242(x, h, wx, wh, bias):
    """Implement one GRU step with gates ordered reset, update, candidate. Use
    reset-before-matmul: candidate=tanh(xWn+(r*h)Un+bn), output=z*h+(1-z)*candidate.
    Weights concatenate three gate matrices.
    """
    xr, xz, xn = np.split(np.asarray(wx), 3, axis=1)
    ur, uz, un = np.split(np.asarray(wh), 3, axis=1)
    br, bz, bn = np.split(bias, 3)
    r = expit(x @ xr + h @ ur + br)
    z = expit(x @ xz + h @ uz + bz)
    candidate = np.tanh(x @ xn + r * h @ un + bn)
    return z * h + (1 - z) * candidate
