import numpy as np


def q030(x, y):
    """Return x[i] - y[j] for row vectors; output shape is (N, M, D)."""
    # Singleton axes pair every row in x with every row in y; inputs are not tiled.
    return np.asarray(x)[:, None, :] - np.asarray(y)[None, :, :]
