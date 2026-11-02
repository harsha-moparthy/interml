import numpy as np


def q153(labels, count):
    """Predict the most frequent training class count times, breaking ties by sorted class
    order.
    """
    classes, counts = np.unique(labels, return_counts=True)
    return np.repeat(classes[counts.argmax()], count)
