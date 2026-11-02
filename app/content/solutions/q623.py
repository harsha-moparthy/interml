import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def max_pool2d(x, kernel=2, stride=None):
    """Return pooled NCHW tensor and backward cache, without padding.

    First row-major maximum wins ties. Overlapping windows must accumulate gradients.
    """
    stride = kernel if stride is None else stride
    x = np.asarray(x)
    if x.ndim != 4 or kernel < 1 or stride < 1:
        raise ValueError("expected NCHW and positive kernel/stride")
    windows = sliding_window_view(x, (kernel, kernel), axis=(-2, -1))[
        :, :, ::stride, ::stride
    ]
    windows = windows.reshape(*windows.shape[:4], kernel * kernel)
    winners = windows.argmax(-1)
    output = np.take_along_axis(windows, winners[..., None], -1)[..., 0]
    return output, (x.shape, kernel, stride, winners)


def max_pool2d_backward(upstream, cache):
    shape, kernel, stride, winners = cache
    upstream = np.asarray(upstream)
    if upstream.shape != winners.shape:
        raise ValueError("upstream must match pooled output")
    batch, channel, row, column = np.indices(winners.shape)
    dx = np.zeros(shape, dtype=upstream.dtype)
    np.add.at(
        dx,
        (
            batch,
            channel,
            row * stride + winners // kernel,
            column * stride + winners % kernel,
        ),
        upstream,
    )
    return dx
