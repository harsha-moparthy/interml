"""Multi-head attention over projected (sequence_length, model_width) arrays.

The public helper names match the original exercise. Each head attends
independently; concatenating heads restores the original feature width.
"""

import numpy as np


def compute_qkv(X, W_q, W_k, W_v):
    """Project token features into queries, keys, and values."""
    return X @ W_q, X @ W_k, X @ W_v


def self_attention(Q, K, V):
    """Compute stable attention for one head, or a batch of heads.

    Q/K share head width; K/V share sequence length. Empty key sequences are
    invalid. No mask or dropout is applied in this exercise.
    """
    queries, keys, values = (np.asarray(array) for array in (Q, K, V))
    if queries.shape[-1] == 0 or keys.shape[-2] == 0:
        raise ValueError("head width and key sequence must be nonempty")
    scores = queries @ keys.swapaxes(-2, -1) / np.sqrt(queries.shape[-1])
    # Compute exponentials once; subtracting a constant preserves softmax.
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ values


def multi_head_attention(Q, K, V, n_heads):
    """Return (query_length, model_width) context from n_heads attention heads.

    Time: O(query_length * key_length * model_width).
    Score storage: O(n_heads * query_length * key_length).
    """
    queries, keys, values = (np.asarray(array) for array in (Q, K, V))
    if any(array.ndim != 2 for array in (queries, keys, values)):
        raise ValueError("Q, K, and V must be matrices")
    model_dim = queries.shape[1]
    if (
        not isinstance(n_heads, (int, np.integer))
        or n_heads <= 0
        or model_dim % n_heads
    ):
        raise ValueError("n_heads must be positive and divide model width")
    if keys.shape[1] != model_dim or values.shape != keys.shape:
        raise ValueError("Q/K/V widths and K/V sequence lengths must match")
    head_dim = model_dim // n_heads

    def split_heads(array):
        return array.reshape(len(array), n_heads, head_dim).transpose(1, 0, 2)

    # A batched matmul computes every head; no per-head Python loop is needed.
    context = self_attention(
        split_heads(queries), split_heads(keys), split_heads(values)
    )
    return context.transpose(1, 0, 2).reshape(len(queries), model_dim)
