import numpy as np


def q262(image):
    """Compute Sobel horizontal and vertical gradients and their magnitude for a grayscale
    image, using edge-replicated padding.
    """
    a = np.pad(np.asarray(image, float), 1, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(a, (3, 3))
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gx = np.einsum("ijab,ab->ij", windows, kernel)
    gy = np.einsum("ijab,ab->ij", windows, kernel.T)
    return (gx, gy, np.hypot(gx, gy))
