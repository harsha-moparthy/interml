import numpy as np


def q174(x, k, seed=0):
    """Choose k initial centers with k-means++: sample the first point uniformly and later
    points proportionally to squared distance to the closest chosen center. Fall back to
    an unused point if all distances vanish.
    """
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    if not 1 <= k <= len(x):
        raise ValueError("invalid k")
    chosen = [int(rng.integers(len(x)))]
    distance = np.full(len(x), np.inf)
    while len(chosen) < k:
        distance = np.minimum(distance, ((x - x[chosen[-1]]) ** 2).sum(1))
        if distance.sum() > 0:
            index = int(rng.choice(len(x), p=distance / distance.sum()))
        else:
            index = int(rng.choice(np.setdiff1d(np.arange(len(x)), chosen)))
        chosen.append(index)
    return x[chosen].copy()
