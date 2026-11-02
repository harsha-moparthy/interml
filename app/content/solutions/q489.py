import numpy as np


def q489(ratings, user, item):
    """Predict one missing rating using mean-centered user-based collaborative filtering.
    Missing ratings are NaN. Use cosine similarity on common observed items; normalize
    by absolute weights and fall back to the user's mean.
    """
    r = np.asarray(ratings, float)
    means = np.nanmean(r, axis=1)
    means = np.where(np.isnan(means), np.nanmean(r), means)
    numerator = denominator = 0.0
    for v in range(len(r)):
        if v == user or np.isnan(r[v, item]):
            continue
        common = np.isfinite(r[user]) & np.isfinite(r[v])
        a, b = (r[user, common] - means[user], r[v, common] - means[v])
        scale = np.linalg.norm(a) * np.linalg.norm(b)
        weight = float(a @ b / scale) if scale else 0.0
        numerator += weight * (r[v, item] - means[v])
        denominator += abs(weight)
    return float(means[user] + (numerator / denominator if denominator else 0))
