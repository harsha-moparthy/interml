import numpy as np


def q157(predictions, true_function):
    """Given bootstrap model predictions of shape (models,points) and noiseless truth per
    point, return mean squared bias, prediction variance, and their sum.
    """
    p = np.asarray(predictions, float)
    mean = p.mean(0)
    bias = np.mean((mean - true_function) ** 2)
    variance = np.mean(p.var(0))
    return (float(bias), float(variance), float(bias + variance))
