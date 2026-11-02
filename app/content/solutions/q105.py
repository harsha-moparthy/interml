import numpy as np


def q105(probabilities):
    """Given a softmax probability vector p, return its Jacobian diag(p)-p*p.T."""
    p = np.asarray(probabilities, float)
    return np.diag(p) - np.outer(p, p)
