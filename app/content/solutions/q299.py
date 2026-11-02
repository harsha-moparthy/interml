import numpy as np


def q299(counts, alpha=1):
    """Apply additive smoothing to a bigram count matrix and normalize each row into a
    next-token probability distribution.
    """
    values = np.asarray(counts, float) + alpha
    if np.any(values.sum(1) == 0):
        raise ValueError("zero-probability row")
    return values / values.sum(axis=1, keepdims=True)
