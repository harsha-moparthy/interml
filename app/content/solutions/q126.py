import numpy as np


def q126(values, period):
    """Encode cyclic numeric values as sine and cosine columns with a positive period.
    Return shape (n,2).
    """
    if period <= 0:
        raise ValueError("period must be positive")
    angle = 2 * np.pi * np.asarray(values) / period
    return np.column_stack([np.sin(angle), np.cos(angle)])
