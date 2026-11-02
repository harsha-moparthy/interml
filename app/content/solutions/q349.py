import numpy as np


def q349(hidden, unembedding, scale, bias, epsilon=1e-05):
    """Apply a logit lens: normalize intermediate hidden states with an affine LayerNorm,
    then project through an unembedding matrix of shape (D,V).
    """
    x = np.asarray(hidden, float)
    normalized = (x - x.mean(-1, keepdims=True)) / np.sqrt(
        x.var(-1, keepdims=True) + epsilon
    )
    return (normalized * scale + bias) @ unembedding
