import numpy as np


def pca(x, components, *, method="svd"):
    """Center data and return transformed rows, principal axes, variance, and mean.

    Axes are rows, sorted by decreasing variance. Signs are conventional and can
    differ between methods; compare subspaces/reconstructions rather than signs.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) < 2 or not 1 <= components <= min(x.shape):
        raise ValueError("invalid PCA dimensions")
    mean = x.mean(0)
    centered = x - mean
    if method == "svd":
        _, singular, axes = np.linalg.svd(centered, full_matrices=False)
        variance = singular**2 / (len(x) - 1)
    elif method == "eigh":
        variance, vectors = np.linalg.eigh(centered.T @ centered / (len(x) - 1))
        order = np.argsort(variance)[::-1]
        axes, variance = vectors[:, order].T, np.maximum(variance[order], 0)
    else:
        raise ValueError("method must be svd or eigh")
    axes = axes[:components]
    return centered @ axes.T, axes, variance[:components], mean
