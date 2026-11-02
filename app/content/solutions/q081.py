"""Calculate a correlation matrix for X, optionally including variables from Y."""

import numpy as np


def calculate_correlation_matrix(X, Y=None):

    def calculate_std_dev(A):
        return np.sqrt(np.mean((A - A.mean(0)) ** 2, axis=0))

    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    covariance = 1 / n_samples * (X - X.mean(0)).T.dot(Y - Y.mean(0))
    std_dev_X = np.expand_dims(calculate_std_dev(X), 1)
    std_dev_y = np.expand_dims(calculate_std_dev(Y), 1)
    correlation_matrix = np.divide(covariance, std_dev_X.dot(std_dev_y.T))
    return np.array(correlation_matrix, dtype=float)
