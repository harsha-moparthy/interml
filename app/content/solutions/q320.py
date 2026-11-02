import numpy as np


def q309(query, key, value, allowed=None):
    queries, keys, values = (
        np.asarray(query, float),
        np.asarray(key, float),
        np.asarray(value, float),
    )
    scores = queries @ keys.swapaxes(-1, -2) / np.sqrt(queries.shape[-1])
    valid = (
        np.ones_like(scores, bool)
        if allowed is None
        else np.broadcast_to(allowed, scores.shape)
    )
    scores = np.where(valid, scores, -np.inf)
    maximum = np.max(scores, axis=-1, keepdims=True)
    # An all-masked row has maximum -inf; use zero so subtracting it stays defined.
    maximum = np.where(np.isfinite(maximum), maximum, 0)
    weights = np.where(valid, np.exp(scores - maximum), 0)
    denominator = weights.sum(axis=-1, keepdims=True)
    weights = np.divide(
        weights,
        denominator,
        out=np.zeros_like(weights),
        where=denominator > 0,
    )
    return weights @ values


def q320(query, key, value, allowed=None):
    """Implement grouped-query attention. Q has Hq heads, K/V have Hkv heads, and Hq must
    be divisible by Hkv. Consecutive query-head groups share one KV head.
    """
    queries, keys, values = (np.asarray(query), np.asarray(key), np.asarray(value))
    if queries.shape[1] % keys.shape[1]:
        raise ValueError("incompatible head counts")
    # Consecutive query heads share one KV head; repeat materializes those heads.
    repeats = queries.shape[1] // keys.shape[1]
    return q309(
        queries,
        np.repeat(keys, repeats, axis=1),
        np.repeat(values, repeats, axis=1),
        allowed,
    )
