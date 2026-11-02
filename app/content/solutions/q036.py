import numpy as np


def q036(matrix):
    """Return each row’s minimum, maximum, and first maximum’s column index."""
    x = np.asarray(matrix)
    return (x.min(axis=1), x.max(axis=1), x.argmax(axis=1))
