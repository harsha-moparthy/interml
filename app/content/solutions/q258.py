import numpy as np


def q258(image):
    """Convert an RGB image with channels last to floating-point grayscale using
    coefficients 0.299, 0.587, 0.114.
    """
    return np.asarray(image, float) @ np.array([0.299, 0.587, 0.114])
