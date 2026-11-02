import numpy as np


def kmeans(x, clusters, *, iterations=100, seed=0, tolerance=1e-6):
    """K-means++ followed by Lloyd updates; return centers and final nearest labels.

    Empty clusters are reseeded from points with the largest current error.
    Duplicate data may make some clusters permanently empty; output remains finite.
    """
    x = np.asarray(x, dtype=float)
    if (
        x.ndim != 2
        or not 1 <= clusters <= len(x)
        or iterations < 1
        or not np.isfinite(x).all()
    ):
        raise ValueError("invalid data, cluster count, or iteration budget")
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))].copy()]
    for _ in range(1, clusters):
        distances = ((x[:, None] - np.asarray(centers)[None, :]) ** 2).sum(-1).min(-1)
        index = (
            rng.choice(len(x), p=distances / distances.sum())
            if distances.sum() > 0
            else rng.integers(len(x))
        )
        centers.append(x[index].copy())
    centers = np.asarray(centers)
    for _ in range(iterations):
        distances = ((x[:, None] - centers[None, :]) ** 2).sum(-1)
        labels = distances.argmin(-1)
        counts = np.bincount(labels, minlength=clusters)
        sums = np.zeros_like(centers)
        np.add.at(sums, labels, x)
        updated = sums / np.maximum(counts[:, None], 1)
        empty = np.flatnonzero(counts == 0)
        farthest = np.argsort(-distances.min(-1), kind="stable")
        updated[empty] = x[farthest[: len(empty)]]
        converged = np.max(np.abs(updated - centers)) <= tolerance
        centers = updated
        if converged:
            break
    labels = ((x[:, None] - centers[None, :]) ** 2).sum(-1).argmin(-1)
    return centers, labels
