import numpy as np


def q119(x):
    """Robust-scale each column by subtracting its median and dividing by its interquartile
    range. Use scale one for constant columns.
    """
    x = np.asarray(x, float)
    q1, median, q3 = np.quantile(x, [0.25, 0.5, 0.75], axis=0)
    scale = np.where(q3 > q1, q3 - q1, 1)
    return (x - median) / scale
