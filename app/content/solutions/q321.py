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


def q321(query, latent, w_key, w_value, allowed=None):
    """Implement the compressed-KV core of multi-head latent attention. Latent cache is
    (B,S,R), per-head key/value up-projections are (H,R,D)/(H,R,Dv), and Q is (B,H,L,D).
    This variant omits the separate RoPE branch.
    """
    keys = np.einsum("bsr,hrd->bhsd", latent, w_key)
    values = np.einsum("bsr,hrv->bhsv", latent, w_value)
    return q309(query, keys, values, allowed)
