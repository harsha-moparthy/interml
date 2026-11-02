import numpy as np


def q186(x, rank, oversampling=5, iterations=2, seed=0):
    """Compute randomized PCA: center X, sketch its range with Gaussian vectors and power
    iterations, then SVD the projected matrix. Return mean, component rows, and retained
    singular values.
    """
    x = np.asarray(x, float)
    mean = x.mean(0)
    a = x - mean
    rng = np.random.default_rng(seed)
    omega = rng.normal(size=(a.shape[1], min(rank + oversampling, min(a.shape))))
    q, _ = np.linalg.qr(a @ omega, mode="reduced")
    for _ in range(iterations):
        q, _ = np.linalg.qr(a @ (a.T @ q), mode="reduced")
    _, s, vh = np.linalg.svd(q.T @ a, full_matrices=False)
    return (mean, vh[:rank], s[:rank])
