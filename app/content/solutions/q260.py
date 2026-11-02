import numpy as np


def q260(image, padding):
    """Zero-pad the height and width of an HWC image equally on all sides without padding
    channels.
    """
    return np.pad(image, ((padding, padding), (padding, padding), (0, 0)))
