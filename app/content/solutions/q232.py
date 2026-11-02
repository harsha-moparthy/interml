"""Normalize BCHW feature maps over batch and spatial axes, then apply channel scale and bias."""

import numpy as np


def batch_normalization(
    X: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    epsilon: float = 1e-05,
) -> np.ndarray:
    mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
    variance = np.var(X, axis=(0, 2, 3), keepdims=True)
    X_norm = (X - mean) / np.sqrt(variance + epsilon)
    norm_X = gamma * X_norm + beta
    return norm_X
