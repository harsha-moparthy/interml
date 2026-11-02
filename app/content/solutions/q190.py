import numpy as np


def q190(x, labels):
    """Compute mean silhouette score with Euclidean distances for at least two clusters.
    Define singleton-cluster samples to have silhouette zero.
    """
    x, labels = (np.asarray(x, float), np.asarray(labels))
    distance = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    scores = []
    for i in range(len(x)):
        own = labels == labels[i]
        own[i] = False
        if not own.any():
            scores.append(0)
            continue
        a = distance[i, own].mean()
        b = min(
            (
                distance[i, labels == c].mean()
                for c in np.unique(labels)
                if c != labels[i]
            )
        )
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0)
    return float(np.mean(scores))
