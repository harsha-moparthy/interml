import numpy as np


def q116(train, test):
    """Fit column means and population standard deviations on training data only, transform
    both matrices, and return transformed data plus fitted statistics. Replace zero
    scales with one.
    """
    train, test = (np.asarray(train, float), np.asarray(test, float))
    mean = train.mean(0)
    scale = train.std(0)
    scale = np.where(scale == 0, 1, scale)
    return ((train - mean) / scale, (test - mean) / scale, mean, scale)
