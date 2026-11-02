import numpy as np


def q037(vector, repeats):
    """Repeat a vector, then take adjacent differences including repetition boundaries."""
    return np.diff(np.tile(vector, repeats))
