import numpy as np


def q026(array, mask):
    """Return a flat selection using a boolean mask with exactly the input shape."""
    array, mask = (np.asarray(array), np.asarray(mask, dtype=bool))
    if array.shape != mask.shape:
        raise ValueError("shape mismatch")
    return array[mask]
