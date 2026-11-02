"""Apply a two-layer residual block with an identity shortcut."""

import numpy as np


def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    y = np.dot(w1, x)
    y = np.maximum(0, y)
    y = np.dot(w2, y)
    y = y + x
    y = np.maximum(0, y)
    return y
