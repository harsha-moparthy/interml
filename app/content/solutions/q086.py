import numpy as np


def q086(joint, column):
    """For a nonnegative joint probability table P(X,Y), return P(X|Y=column). Reject a
    zero-probability conditioning event.
    """
    p = np.asarray(joint, float)[:, column]
    if p.sum() <= 0:
        raise ValueError("impossible conditioning event")
    return p / p.sum()
