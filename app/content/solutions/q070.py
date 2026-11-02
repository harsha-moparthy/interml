import numpy as np


def q070(matrix, tolerance=1e-10):
    """Return an orthonormal basis for a matrix's null space as columns. Use full SVD and
    an absolute singular-value threshold.
    """
    _, singular, vh = np.linalg.svd(np.asarray(matrix, float), full_matrices=True)
    rank = np.count_nonzero(singular > tolerance)
    return vh[rank:].T
