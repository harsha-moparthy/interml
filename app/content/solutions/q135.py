"""Fit linear regression by gradient descent; X includes the intercept column."""

import numpy as np


def linear_regression_gradient_descent(
    X: np.ndarray, y: np.ndarray, alpha: float, iterations: int
) -> np.ndarray:
    m, n = X.shape
    y = y.reshape(-1, 1)
    theta = np.zeros((n, 1))
    for _ in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = alpha * (X.T @ errors) / m
        theta -= gradient
    return np.round(theta.flatten(), 4)
