import numpy as np


def q022(start, stop, step, count):
    """Return a half-open stepped range and an endpoint-inclusive evenly spaced grid."""
    if step == 0 or count < 0:
        raise ValueError("invalid step or count")
    return (np.arange(start, stop, step), np.linspace(start, stop, count))
