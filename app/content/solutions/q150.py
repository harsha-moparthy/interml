import numpy as np


def q150(scores, labels):
    """Return mean binary hinge loss for labels in {-1,+1} and unrestricted decision
    scores.
    """
    return float(np.maximum(0, 1 - np.asarray(labels) * scores).mean())
