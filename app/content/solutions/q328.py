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


def q328(decoder, encoder, wq, wk, wv, wo, heads, source_mask=None):
    """Implement multi-head cross-attention with decoder queries and encoder keys/values.
    Decoder is (B,L,D), encoder is (B,S,D); L and S may differ. Source mask is (B,S),
    True for valid tokens. Do not apply a causal mask.
    """
    b, l, d = decoder.shape
    s = encoder.shape[1]
    if d % heads:
        raise ValueError("incompatible heads")
    q = (decoder @ wq).reshape(b, l, heads, d // heads).transpose(0, 2, 1, 3)
    k = (encoder @ wk).reshape(b, s, heads, d // heads).transpose(0, 2, 1, 3)
    v = (encoder @ wv).reshape(b, s, heads, d // heads).transpose(0, 2, 1, 3)
    allowed = None if source_mask is None else np.asarray(source_mask)[:, None, None, :]
    return q309(q, k, v, allowed).transpose(0, 2, 1, 3).reshape(b, l, d) @ wo
