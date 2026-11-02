import numpy as np


def q031(matrix, scales):
    """Multiply each matrix column by its corresponding scale using broadcasting."""
    return np.asarray(matrix) * np.asarray(scales)[None, :]
