"""Compute mean absolute error between true and predicted values."""

import numpy as np


def mae(y_true, y_pred):
    if y_true.shape != y_pred.shape:
        raise ValueError("Arrays must have the same shape")
    if y_true.size == 0:
        raise ValueError("Arrays cannot be empty")
    return round(np.mean(np.abs(y_true - y_pred)), 3)
