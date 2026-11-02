import numpy as np


def q029(matrix, scales):
    """Multiply each matrix row by its corresponding scale using broadcasting."""
    return np.asarray(matrix) * np.asarray(scales)[:, None]
