import numpy as np


def q183(x, components, gamma=1):
    """Compute training-set kernel PCA coordinates with an RBF kernel. Double-center the
    Gram matrix and keep its largest nonnegative eigenvalues.
    """
    x = np.asarray(x, float)
    distance = ((x[:, None] - x[None, :]) ** 2).sum(-1)
    kernel = np.exp(-gamma * distance)
    centered = (
        kernel - kernel.mean(0)[None, :] - kernel.mean(1)[:, None] + kernel.mean()
    )
    values, vectors = np.linalg.eigh(centered)
    order = np.argsort(values)[::-1][:components]
    return vectors[:, order] * np.sqrt(np.maximum(values[order], 0))
