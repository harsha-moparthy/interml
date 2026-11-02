import numpy as np


def q234(x, scale, epsilon=1e-06):
    """Apply RMSNorm along the last dimension, without subtracting the mean, then multiply
    by a learned scale vector.
    """
    x = np.asarray(x, float)
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon) * scale
