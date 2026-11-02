import numpy as np


def q178(affinity, k, seed=0):
    """Perform normalized spectral clustering on a symmetric nonnegative affinity matrix
    with positive degrees. Compute the k smallest Laplacian eigenvectors, normalize
    rows, and use sklearn KMeans for the final discretization.
    """
    from sklearn.cluster import KMeans

    a = np.asarray(affinity, float)
    degree = a.sum(1)
    if np.any(degree <= 0):
        raise ValueError("isolated nodes")
    inv = 1 / np.sqrt(degree)
    laplacian = np.eye(len(a)) - inv[:, None] * a * inv[None, :]
    _, vectors = np.linalg.eigh(laplacian)
    embedding = vectors[:, :k]
    embedding /= np.maximum(np.linalg.norm(embedding, axis=1, keepdims=True), 1e-12)
    return KMeans(k, n_init=10, random_state=seed).fit_predict(embedding)
