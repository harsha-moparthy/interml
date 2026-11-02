import numpy as np
from scipy.special import expit


def q345(logits, bias, k):
    """Select MoE experts using sigmoid affinity plus a load-correction bias. Return top-k
    expert indices and normalized original sigmoid affinities; the bias affects
    selection only.
    """
    affinity = expit(np.asarray(logits, float))
    index = np.argsort(-(affinity + bias), axis=-1, kind="stable")[..., :k]
    weights = np.take_along_axis(affinity, index, axis=-1)
    weights /= weights.sum(-1, keepdims=True)
    return (index, weights)
