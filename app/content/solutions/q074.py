import numpy as np


def q074(matrix):
    """Compute the lower-triangular Cholesky factor of a symmetric positive-definite matrix
    using scalar updates.
    """
    a = np.asarray(matrix, float)
    l = np.zeros_like(a)
    if not np.allclose(a, a.T):
        raise ValueError("not symmetric")
    for i in range(len(a)):
        for j in range(i + 1):
            value = a[i, j] - l[i, :j] @ l[j, :j]
            if i == j:
                if value <= 0:
                    raise ValueError("not positive definite")
                l[i, j] = np.sqrt(value)
            else:
                l[i, j] = value / l[j, j]
    return l
