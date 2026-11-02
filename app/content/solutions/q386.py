import numpy as np
from scipy.special import logsumexp


def q386(logits, p=0.9, seed=0):
    """Sample from the smallest descending-probability prefix whose mass reaches p,
    including the threshold-crossing token. Require 0<p<=1.
    """
    if not 0 < p <= 1:
        raise ValueError("invalid mass")
    x = np.asarray(logits, float)
    order = np.argsort(-x, kind="stable")
    prob = np.exp(x[order] - logsumexp(x))
    count = min(len(x), np.searchsorted(np.cumsum(prob), p) + 1)
    prob = prob[:count]
    prob /= prob.sum()
    return int(np.random.default_rng(seed).choice(order[:count], p=prob))
