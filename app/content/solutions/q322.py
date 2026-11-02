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


def q322(query, key, value, window):
    """Implement causal sliding-window self-attention over matching sequence lengths. Each
    position sees itself and at most window-1 preceding positions.
    """
    if window <= 0:
        raise ValueError("positive window required")
    length = np.asarray(query).shape[-2]
    distance = np.arange(length)[:, None] - np.arange(length)[None, :]
    return q309(query, key, value, (distance >= 0) & (distance < window))
