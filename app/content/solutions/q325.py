import numpy as np


def q325(query, key, value, split_size=64):
    """Implement split-K decoding attention for one query (D,) against keys (S,D). Compute
    per-split maxima, normalizers, and weighted values, then merge them exactly. This is
    the per-head core used by FlashDecoding.
    """
    query, key, value = (
        np.asarray(query, float),
        np.asarray(key, float),
        np.asarray(value, float),
    )
    partials = []
    for start in range(0, len(key), split_size):
        scores = key[start : start + split_size] @ query / np.sqrt(len(query))
        maximum = scores.max()
        p = np.exp(scores - maximum)
        partials.append((maximum, p.sum(), p @ value[start : start + split_size]))
    maximum = max((x[0] for x in partials))
    numerator = np.zeros(value.shape[1])
    denominator = 0.0
    for m, z, out in partials:
        scale = np.exp(m - maximum)
        numerator += scale * out
        denominator += scale * z
    return numerator / denominator
