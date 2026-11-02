import numpy as np


def q391(cache, new_keys, new_values):
    """Append projected K/V arrays of shape (B,H,T,D) to an optional (K,V) cache along the
    sequence axis. Return a new cache without modifying the inputs.
    """
    # Copy the initial cache so later changes cannot alias the caller’s arrays.
    if cache is None:
        return (
            np.array(new_keys, copy=True),
            np.array(new_values, copy=True),
        )
    return (
        np.concatenate((cache[0], new_keys), axis=-2),
        np.concatenate((cache[1], new_values), axis=-2),
    )
