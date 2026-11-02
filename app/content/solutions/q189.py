import numpy as np


def q189(x, k=5):
    """Compute fixed-k Local Outlier Factor scores from Euclidean distances. Use exactly k
    nearest other points, with deterministic sort-based tie handling and a small
    reachability floor.
    """
    x = np.asarray(x, float)
    distance = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    np.fill_diagonal(distance, np.inf)
    neighbors = np.argsort(distance, axis=1)[:, :k]
    kth = np.take_along_axis(distance, neighbors, axis=1)[:, -1]
    reach = np.maximum(np.take_along_axis(distance, neighbors, axis=1), kth[neighbors])
    density = 1 / np.maximum(reach.mean(1), 1e-12)
    return density[neighbors].mean(1) / density
