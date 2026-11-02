import numpy as np


def q073(matrix):
    """Compute reduced QR using modified Gram-Schmidt for a full-column-rank tall matrix.
    Return orthonormal Q and upper-triangular R.
    """
    a = np.asarray(matrix, float)
    q = np.zeros_like(a)
    r = np.zeros((a.shape[1], a.shape[1]))
    for j in range(a.shape[1]):
        v = a[:, j].copy()
        for i in range(j):
            r[i, j] = q[:, i] @ v
            v -= r[i, j] * q[:, i]
        r[j, j] = np.linalg.norm(v)
        if r[j, j] < 1e-12:
            raise ValueError("dependent columns")
        q[:, j] = v / r[j, j]
    return (q, r)
