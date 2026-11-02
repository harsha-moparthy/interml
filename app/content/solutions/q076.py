import numpy as np


def q076(matrix):
    """Compute a full SVD of a real 2x2 matrix, including singular matrices. Use a
    symmetric eigensolver on A.T@A; return U, descending singular values, and V.T with
    orthogonal factors.
    """
    a = np.asarray(matrix, float)
    if a.shape != (2, 2):
        raise ValueError("2x2 matrix required")
    eigen, v = np.linalg.eigh(a.T @ a)
    order = np.argsort(eigen)[::-1]
    v = v[:, order]
    singular = np.sqrt(np.maximum(eigen[order], 0))
    tolerance = np.finfo(float).eps * max(1, singular[0]) * 10
    if singular[0] <= tolerance:
        return (np.eye(2), np.zeros(2), v.T)
    first = a @ v[:, 0]
    first /= np.linalg.norm(first)
    second = np.array([-first[1], first[0]])
    if singular[1] > tolerance and np.dot(second, a @ v[:, 1]) < 0:
        second = -second
    return (np.column_stack([first, second]), singular, v.T)
