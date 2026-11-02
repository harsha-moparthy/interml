import numpy as np


def q312(length):
    """Construct a Boolean lower-triangular causal permission mask of shape
    (length,length), including the diagonal.
    """
    return np.arange(length)[:, None] >= np.arange(length)[None, :]
