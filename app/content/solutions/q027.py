import numpy as np


def q027(array):
    """Demonstrate that basic slicing aliases data while advanced indexing copies it."""
    a = np.array(array, copy=True)
    b = np.array(array, copy=True)
    view = a[::2]
    view[:] = -1
    # Integer-array indexing allocates new storage; this write cannot affect b.
    copied = b[np.arange(0, len(b), 2)]
    copied[:] = -1
    return (a, b)
