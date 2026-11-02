"""Normalize groups of channels within each BCHW example and apply affine parameters."""

import numpy as np


def group_normalization(
    X: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    num_groups: int,
    epsilon: float = 1e-05,
) -> np.ndarray:
    batch_size, num_channels, height, width = X.shape
    group_size = num_channels // num_groups
    X_reshaped = X.reshape(batch_size, num_groups, group_size, height, width)
    mean = np.mean(X_reshaped, axis=(2, 3, 4), keepdims=True)
    variance = np.var(X_reshaped, axis=(2, 3, 4), keepdims=True)
    X_norm = (X_reshaped - mean) / np.sqrt(variance + epsilon)
    X_norm = X_norm.reshape(batch_size, num_channels, height, width)
    norm_X = gamma * X_norm + beta
    return norm_X
