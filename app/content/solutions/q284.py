import numpy as np


def q284(x, y, rectangle, permutation):
    """Apply CutMix to NCHW images using a supplied integer rectangle (x1,y1,x2,y2) and
    batch permutation. Clip the rectangle to the image and mix one-hot labels by actual
    pasted area.
    """
    output = x.clone()
    h, w = x.shape[-2:]
    x1, y1, x2, y2 = rectangle
    x1, x2 = np.clip([x1, x2], 0, w)
    y1, y2 = np.clip([y1, y2], 0, h)
    output[:, :, y1:y2, x1:x2] = x[permutation, :, y1:y2, x1:x2]
    fraction = max(0, x2 - x1) * max(0, y2 - y1) / (h * w)
    return (output, (1 - fraction) * y + fraction * y[permutation])
