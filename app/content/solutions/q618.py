import numpy as np


def online_softmax(logits, block_size=128):
    """Stable softmax of a finite nonempty vector with online normalization statistics.

    The first pass merges block maxima/sums without a full-vector max operation.
    Producing normalized individual probabilities requires a second pass (or
    retaining/rescaling all earlier outputs). Time O(N), auxiliary space O(block).
    FlashAttention instead accumulates weighted values online to avoid N-by-N output.
    """
    x = np.asarray(logits, dtype=float)
    if x.ndim != 1 or not len(x) or block_size < 1 or not np.isfinite(x).all():
        raise ValueError("finite nonempty vector and positive block size required")
    maximum, denominator = -np.inf, 0.0
    for start in range(0, len(x), block_size):
        block = x[start : start + block_size]
        next_maximum = max(maximum, block.max())
        denominator = (
            denominator * np.exp(maximum - next_maximum)
            + np.exp(block - next_maximum).sum()
        )
        maximum = next_maximum
    output = np.empty_like(x)
    for start in range(0, len(x), block_size):
        output[start : start + block_size] = (
            np.exp(x[start : start + block_size] - maximum) / denominator
        )
    return output


def online_attention(query, key, value, block_size=128):
    """One-pass blockwise attention for Q(L,D), K(S,D), V(S,Dv), without an L*S matrix.

    Keep running row maxima, denominators, and weighted-value numerators. Each
    key block is consumed once. This reference has no masking or backward kernel.
    """
    q, k, v = (np.asarray(a, dtype=float) for a in (query, key, value))
    if (
        q.ndim != 2
        or k.ndim != 2
        or v.ndim != 2
        or q.shape[1] != k.shape[1]
        or len(k) != len(v)
        or not len(k)
        or not q.shape[1]
        or block_size < 1
    ):
        raise ValueError("invalid attention dimensions")
    maximum = np.full((len(q), 1), -np.inf)
    denominator = np.zeros((len(q), 1))
    numerator = np.zeros((len(q), v.shape[1]))
    for start in range(0, len(k), block_size):
        scores = q @ k[start : start + block_size].T / np.sqrt(q.shape[1])
        next_maximum = np.maximum(maximum, scores.max(-1, keepdims=True))
        correction = np.exp(maximum - next_maximum)
        weights = np.exp(scores - next_maximum)
        numerator = correction * numerator + weights @ v[start : start + block_size]
        denominator = correction * denominator + weights.sum(-1, keepdims=True)
        maximum = next_maximum
    return numerator / denominator
