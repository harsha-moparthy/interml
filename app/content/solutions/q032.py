import numpy as np


def q032(a, b):
    """Compute (a - b) / (a + b) after broadcasting; return zero where the sum is zero."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    # where prevents division itself; np.where would evaluate both branches.
    numerator = a - b
    denominator = a + b
    return np.divide(
        numerator, denominator, out=np.zeros_like(a), where=denominator != 0
    )
