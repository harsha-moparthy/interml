import numpy as np


def q191(x, labels):
    """Compute the Davies-Bouldin index using mean Euclidean distance to cluster centroids.
    Require at least two clusters with distinct centroids.
    """
    x, labels = (np.asarray(x, float), np.asarray(labels))
    classes = np.unique(labels)
    centers = np.stack([x[labels == c].mean(0) for c in classes])
    scatter = np.array(
        [
            np.linalg.norm(x[labels == c] - m, axis=1).mean()
            for c, m in zip(classes, centers)
        ]
    )
    distance = np.linalg.norm(centers[:, None] - centers[None, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    ratios = (scatter[:, None] + scatter[None, :]) / distance
    return float(ratios.max(axis=1).mean())
