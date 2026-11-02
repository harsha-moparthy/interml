import numpy as np


def q172(model, x, y, score, repeats=5, seed=0):
    """Compute permutation feature importance on held-out data. Score(model,X,y) must be
    larger-is-better. Return mean score decrease per column over repeated shuffles.
    """
    x = np.asarray(x)
    baseline = score(model, x, y)
    rng = np.random.default_rng(seed)
    result = []
    for j in range(x.shape[1]):
        drops = []
        for _ in range(repeats):
            changed = x.copy()
            changed[:, j] = rng.permutation(changed[:, j])
            drops.append(baseline - score(model, changed, y))
        result.append(np.mean(drops))
    return np.array(result)
