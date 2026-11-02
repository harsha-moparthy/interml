import numpy as np


def q261(image, height, width):
    """Resize a two-dimensional image with bilinear interpolation and align-corners
    coordinates. Handle singleton source and destination dimensions.
    """
    a = np.asarray(image, float)
    ys = np.linspace(0, a.shape[0] - 1, height)
    xs = np.linspace(0, a.shape[1] - 1, width)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, a.shape[0] - 1)
    x1 = np.minimum(x0 + 1, a.shape[1] - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    return (1 - wy) * ((1 - wx) * a[np.ix_(y0, x0)] + wx * a[np.ix_(y0, x1)]) + wy * (
        (1 - wx) * a[np.ix_(y1, x0)] + wx * a[np.ix_(y1, x1)]
    )
