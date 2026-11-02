import numpy as np


def q324(query, key, value, block=32):
    """Implement an exact dense attention reference using online softmax over key blocks,
    without storing the full score matrix. Q is (L,D), K is (S,D), V is (S,Dv); use
    float64 accumulators.
    """
    q, k, v = (
        np.asarray(query, float),
        np.asarray(key, float),
        np.asarray(value, float),
    )
    maximum = np.full((len(q), 1), -np.inf)
    normalizer = np.zeros((len(q), 1))
    output = np.zeros((len(q), v.shape[1]))
    for start in range(0, len(k), block):
        scores = q @ k[start : start + block].T / np.sqrt(q.shape[1])
        new_max = np.maximum(maximum, scores.max(1, keepdims=True))
        correction = np.exp(maximum - new_max)
        probabilities = np.exp(scores - new_max)
        output = output * correction + probabilities @ v[start : start + block]
        normalizer = normalizer * correction + probabilities.sum(1, keepdims=True)
        maximum = new_max
    return np.divide(
        output, normalizer, out=np.zeros_like(output), where=normalizer > 0
    )
