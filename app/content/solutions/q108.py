import numpy as np


def q108(gradient, original_shape):
    """Reduce an upstream NumPy gradient to an input shape that was broadcast in the
    forward pass. Sum inserted axes and singleton dimensions.
    """
    g = np.asarray(gradient)
    # Broadcasted leading axes are reduced away before singleton axes are summed.
    while g.ndim > len(original_shape):
        g = g.sum(axis=0)
    axes = tuple((i for i, n in enumerate(original_shape) if n == 1))
    return (
        g.sum(axis=axes, keepdims=True).reshape(original_shape)
        if axes
        else g.reshape(original_shape)
    )
