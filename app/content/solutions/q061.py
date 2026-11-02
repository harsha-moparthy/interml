import numpy as np


def q061(x):
    """Return L1, L2, and infinity norms of the flattened input, plus its Frobenius norm
    when viewed as a matrix.
    """
    x = np.asarray(x, float)
    flat = x.ravel()
    return (
        np.abs(flat).sum(),
        np.sqrt(flat @ flat),
        np.abs(flat).max(initial=0),
        np.sqrt((x * x).sum()),
    )
