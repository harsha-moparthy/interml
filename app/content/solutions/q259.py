import numpy as np


def q259(image, horizontal=True):
    """Flip an HWC image horizontally or vertically and return an independent array."""
    return np.flip(image, axis=1 if horizontal else 0).copy()
