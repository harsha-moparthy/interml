import numpy as np


def q484(original, reconstructed, data_range=1):
    """Compute PSNR in decibels from mean squared error and a specified positive signal
    range. Identical arrays return positive infinity.
    """
    mse = np.mean((np.asarray(original, float) - np.asarray(reconstructed, float)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(data_range**2 / mse))
