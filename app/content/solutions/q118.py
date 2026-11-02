import numpy as np


def q118(x, multiplier=1.5):
    """Keep rows for which every feature lies within [Q1-m*IQR,Q3+m*IQR]. Return the
    filtered matrix and row mask.
    """
    x = np.asarray(x, float)
    q1, q3 = np.quantile(x, [0.25, 0.75], axis=0)
    width = q3 - q1
    mask = ((x >= q1 - multiplier * width) & (x <= q3 + multiplier * width)).all(axis=1)
    return (x[mask], mask)
