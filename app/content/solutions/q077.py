import numpy as np


def q077(matrix, rank):
    """Return the best rank-r approximation in Frobenius norm using truncated SVD. Rank may
    be zero and cannot exceed the smaller matrix dimension.
    """
    a = np.asarray(matrix, float)
    if not 0 <= rank <= min(a.shape):
        raise ValueError("invalid rank")
    u, s, vh = np.linalg.svd(a, full_matrices=False)
    return u[:, :rank] * s[:rank] @ vh[:rank]
