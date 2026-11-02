import numpy as np


def q109(gradient, original_shape):
    """Implement backward through a row-major reshape: restore the gradient's original
    input shape without changing element order.
    """
    return np.asarray(gradient).reshape(original_shape)
