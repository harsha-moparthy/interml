import numpy as np


def q490(ratings, user, item):
    """Predict a missing rating by item-based cosine similarity. Compute similarity only on
    co-rated users. Weight the active user's other observed ratings and fall back to
    their mean.
    """
    r = np.asarray(ratings, float)
    numerator = denominator = 0.0
    for j in range(r.shape[1]):
        if j == item or np.isnan(r[user, j]):
            continue
        valid = np.isfinite(r[:, item]) & np.isfinite(r[:, j])
        a, b = (r[valid, item], r[valid, j])
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        weight = float(a @ b / norm) if norm else 0.0
        numerator += weight * r[user, j]
        denominator += abs(weight)
    fallback = np.nanmean(r[user])
    if np.isnan(fallback):
        fallback = np.nanmean(r)
    return float(numerator / denominator if denominator else fallback)
