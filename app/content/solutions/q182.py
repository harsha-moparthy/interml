import numpy as np


def q182(data, components, standardize=True):
    """Fit PCA to real feature rows using a symmetric covariance eigensolver. Optionally
    standardize features, protecting constant columns. Return feature-by-component
    orthonormal loadings, projected rows, and descending eigenvalues; fix signs by each
    loading's largest-magnitude entry.
    """
    x = np.asarray(data, float)
    if len(x) < 2 or not 1 <= components <= x.shape[1]:
        raise ValueError("invalid data or components")
    centered = x - x.mean(0)
    if standardize:
        scale = x.std(0)
        centered /= np.where(scale > 0, scale, 1)
    covariance = centered.T @ centered / (len(x) - 1)
    eigen, vector = np.linalg.eigh(covariance)
    order = np.argsort(eigen)[::-1][:components]
    loading = vector[:, order]
    signs = np.sign(loading[np.argmax(np.abs(loading), axis=0), np.arange(components)])
    loading *= np.where(signs == 0, 1, signs)
    return (loading, centered @ loading, np.maximum(eigen[order], 0))
