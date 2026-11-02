import numpy as np
from scipy.special import logsumexp


def q386(logits, p=0.9, seed=0):
    if not 0 < p <= 1:
        raise ValueError("invalid mass")
    x = np.asarray(logits, float)
    order = np.argsort(-x, kind="stable")
    prob = np.exp(x[order] - logsumexp(x))
    count = min(len(x), np.searchsorted(np.cumsum(prob), p) + 1)
    prob = prob[:count]
    prob /= prob.sum()
    return int(np.random.default_rng(seed).choice(order[:count], p=prob))


def q387(logits, temperature=1, top_k=None, top_p=1, seed=0):
    """Apply temperature, top-k, then nucleus filtering and sample once. temperature=0 is
    greedy; top-k is either None or a positive integer.
    """
    x = np.asarray(logits, float).copy()
    if temperature == 0:
        return int(x.argmax())
    if temperature < 0 or (top_k is not None and top_k < 1):
        raise ValueError("invalid sampling parameter")
    x /= temperature
    if top_k is not None and top_k < len(x):
        keep = np.argsort(-x, kind="stable")[:top_k]
        mask = np.ones(len(x), bool)
        mask[keep] = False
        x[mask] = -np.inf
    return q386(x, top_p, seed)
