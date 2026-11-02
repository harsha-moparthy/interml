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


def q317(x, wqkv, wo, heads, causal=True):
    """Implement multi-head self-attention with one combined QKV projection. X is (B,L,D),
    Wqkv is (D,3D), Wo is (D,D), and D must be divisible by heads.
    """
    x = np.asarray(x, float)
    batch_size, sequence_length, model_dim = x.shape
    if model_dim % heads:
        raise ValueError("head dimension mismatch")
    # Split once after the fused projection, then expose a separate head axis.
    q, k, v = np.split(x @ wqkv, 3, axis=-1)
    q, k, v = [
        projection.reshape(
            batch_size, sequence_length, heads, model_dim // heads
        ).transpose(0, 2, 1, 3)
        for projection in (q, k, v)
    ]
    mask = (
        np.tril(np.ones((sequence_length, sequence_length), bool)) if causal else None
    )
    context = (
        q309(q, k, v, mask)
        .transpose(0, 2, 1, 3)
        .reshape(batch_size, sequence_length, model_dim)
    )
    return context @ wo
