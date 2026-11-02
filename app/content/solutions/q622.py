import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def conv2d(x, weight, bias=None, *, stride=1, padding=0):
    """NCHW cross-correlation with OIHW weights, scalar stride and symmetric padding.

    im2col exposes windows and matmul contracts channel/kernel dimensions. Materialized
    columns cost O(N*OH*OW*C*KH*KW); this is a clear reference, not a low-memory kernel.
    """
    x, weight = np.asarray(x), np.asarray(weight)
    if (
        x.ndim != 4
        or weight.ndim != 4
        or x.shape[1] != weight.shape[1]
        or stride < 1
        or padding < 0
    ):
        raise ValueError("invalid convolution shapes, stride, or padding")
    kh, kw = weight.shape[-2:]
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    windows = sliding_window_view(padded, (kh, kw), axis=(-2, -1))[
        :, :, ::stride, ::stride
    ]
    batch, channels, height, width, _, _ = windows.shape
    columns = windows.transpose(0, 2, 3, 1, 4, 5).reshape(batch * height * width, -1)
    output = columns @ weight.reshape(len(weight), -1).T
    if bias is not None:
        output += np.asarray(bias)
    return output.reshape(batch, height, width, len(weight)).transpose(0, 3, 1, 2)
