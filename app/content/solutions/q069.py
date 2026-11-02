import numpy as np


def q069(matrix, tolerance=None):
    """Compute numerical matrix rank using singular values. Default tolerance is
    max(shape)*machine_epsilon*largest_singular_value.
    """
    a = np.asarray(matrix, float)
    singular = np.linalg.svd(a, compute_uv=False)
    threshold = (
        max(a.shape) * np.finfo(float).eps * singular.max(initial=0)
        if tolerance is None
        else tolerance
    )
    return int(np.count_nonzero(singular > threshold))
