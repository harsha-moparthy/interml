import numpy as np


def q130(minority, count, neighbors=5, seed=0):
    """Generate count SMOTE samples by interpolating between a randomly chosen minority
    point and one of its nearest other minority points. Require at least two points.
    """
    x = np.asarray(minority, float)
    rng = np.random.default_rng(seed)
    if len(x) < 2:
        raise ValueError("at least two minority points required")
    distance = ((x[:, None] - x[None, :]) ** 2).sum(-1)
    np.fill_diagonal(distance, np.inf)
    near = np.argsort(distance, axis=1)[:, : min(neighbors, len(x) - 1)]
    base = rng.integers(len(x), size=count)
    partner = np.array([rng.choice(near[i]) for i in base], dtype=int)
    return x[base] + rng.random((count, 1)) * (x[partner] - x[base])
