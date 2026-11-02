import numpy as np


def q071(matrix):
    """Test whether a real square matrix is symmetric positive definite. Return False for
    nonsymmetric or singular inputs.
    """
    a = np.asarray(matrix, float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or (not np.allclose(a, a.T)):
        return False
    try:
        np.linalg.cholesky(a)
        return True
    except np.linalg.LinAlgError:
        return False
