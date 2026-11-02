import numpy as np


def q025(matrix, rows, columns):
    """Select the Cartesian product of row and column indices, rather than paired entries."""
    # ix_ adds singleton axes so rows and columns form a Cartesian product.
    return np.asarray(matrix)[np.ix_(rows, columns)]
