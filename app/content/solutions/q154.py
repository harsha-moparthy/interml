import numpy as np


def q154(targets, count):
    """Build a constant regression baseline that predicts the training-target mean for
    count test examples.
    """
    return np.full(count, np.mean(targets))
