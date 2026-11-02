import numpy as np


def q149(x, y, degree=3, gamma=1, offset=1):
    """Compute a polynomial kernel matrix (gamma*x_i dot y_j+offset)^degree."""
    return (gamma * np.asarray(x) @ np.asarray(y).T + offset) ** degree
