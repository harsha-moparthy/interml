import numpy as np


def q263(image, size=5, sigma=1):
    """Blur a grayscale image with a normalized square Gaussian kernel. Size must be
    positive and odd; use edge padding.
    """
    if size <= 0 or size % 2 == 0 or sigma <= 0:
        raise ValueError("invalid kernel")
    coordinate = np.arange(size) - size // 2
    kernel = np.exp(
        -(coordinate[:, None] ** 2 + coordinate[None, :] ** 2) / (2 * sigma**2)
    )
    kernel /= kernel.sum()
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(image, size // 2, mode="edge"), (size, size)
    )
    return np.einsum("ijab,ab->ij", windows, kernel)
