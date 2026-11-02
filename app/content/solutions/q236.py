"""Normalize spatial dimensions independently for each example and channel."""

import numpy as np


def instance_normalization(
    X: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    epsilon: float = 1e-05,
) -> np.ndarray:
    gamma = gamma.reshape(1, -1, 1, 1)
    beta = beta.reshape(1, -1, 1, 1)
    mean = np.mean(X, axis=(2, 3), keepdims=True)
    var = np.var(X, axis=(2, 3), keepdims=True)
    X_norm = (X - mean) / np.sqrt(var + epsilon)
    return gamma * X_norm + beta
