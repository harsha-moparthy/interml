import numpy as np


def q039(values, boundaries):
    """Find insertion bins to the right of equal boundaries; require sorted boundaries."""
    boundaries = np.asarray(boundaries)
    if np.any(np.diff(boundaries) < 0):
        raise ValueError("boundaries must be sorted")
    # side="right" assigns a value equal to a boundary to the following bin.
    return np.searchsorted(boundaries, values, side="right")
