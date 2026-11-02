import numpy as np


def q072(matrix):
    """Compute LU factorization with partial pivoting. Return P,L,U satisfying P@A=L@U.
    Reject numerically singular matrices.
    """
    u = np.array(matrix, float, copy=True)
    n = len(u)
    p = np.eye(n)
    l = np.eye(n)
    for k in range(n):
        pivot = k + np.argmax(np.abs(u[k:, k]))
        if abs(u[pivot, k]) < 1e-12:
            raise ValueError("singular matrix")
        u[[k, pivot]] = u[[pivot, k]]
        p[[k, pivot]] = p[[pivot, k]]
        l[[k, pivot], :k] = l[[pivot, k], :k]
        for i in range(k + 1, n):
            l[i, k] = u[i, k] / u[k, k]
            u[i, k:] -= l[i, k] * u[k, k:]
    return (p, l, u)
