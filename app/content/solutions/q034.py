import numpy as np


def q034(values):
    """Return stable sorting indices and sorted values; preserve the order of ties."""
    x = np.asarray(values)
    order = np.argsort(x, kind="stable")
    return (order, x[order])
