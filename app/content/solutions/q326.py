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


def q326(query, key_blocks, value_blocks, block_table, length):
    """Compute attention over a paged KV cache by gathering logical blocks using a block
    table. Block storage is (physical_blocks,block_size,dim); ignore unused positions
    after logical length.
    """
    key = np.asarray(key_blocks)[block_table].reshape(-1, key_blocks.shape[-1])[:length]
    value = np.asarray(value_blocks)[block_table].reshape(-1, value_blocks.shape[-1])[
        :length
    ]
    return q309(query, key, value)
