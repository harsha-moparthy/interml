import numpy as np


def q214(training, serving, tolerance=1e-08):
    """Compare aligned training and serving feature matrices. Return per-column mean
    absolute differences and flags for differences exceeding tolerance. Inputs must be
    finite and equally shaped.
    """
    a, b = (np.asarray(training, float), np.asarray(serving, float))
    if a.shape != b.shape:
        raise ValueError("shape mismatch")
    error = np.abs(a - b).mean(axis=0)
    return (error, error > tolerance)
