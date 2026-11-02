import numpy as np


def q231(anchor, positive, negative, margin=1):
    """Compute mean triplet margin loss with Euclidean distances, max(0,
    distance(anchor,positive)-distance(anchor,negative)+margin).
    """
    a, p, n = (
        np.asarray(anchor),
        np.asarray(positive),
        np.asarray(negative),
    )
    return float(
        np.maximum(
            0,
            np.linalg.norm(a - p, axis=-1) - np.linalg.norm(a - n, axis=-1) + margin,
        ).mean()
    )
