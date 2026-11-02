import numpy as np


def q483(real_features, generated_features):
    """Compute FID from precomputed feature rows using sample covariance. Use the symmetric
    covariance-sandwich eigendecomposition for a real trace square root; require at
    least two samples per set.
    """
    x = np.asarray(real_features, float)
    y = np.asarray(generated_features, float)
    if min(len(x), len(y)) < 2:
        raise ValueError("two samples required")
    a = np.atleast_2d(np.cov(x, rowvar=False))
    b = np.atleast_2d(np.cov(y, rowvar=False))
    eig, v = np.linalg.eigh(a)
    root = v * np.sqrt(np.maximum(eig, 0)) @ v.T
    sandwich = root @ b @ root
    trace_root = np.sqrt(
        np.maximum(np.linalg.eigvalsh((sandwich + sandwich.T) / 2), 0)
    ).sum()
    return float(
        max(
            0,
            np.sum((x.mean(0) - y.mean(0)) ** 2)
            + np.trace(a)
            + np.trace(b)
            - 2 * trace_root,
        )
    )
