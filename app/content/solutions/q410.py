import numpy as np


def q410(weights, channel_axis=0):
    """Apply symmetric INT8 quantization independently to each output channel of a weight
    tensor. Return integer weights and broadcastable scales retaining singleton reduced
    axes.
    """
    w = np.asarray(weights, float)
    axis = channel_axis % w.ndim
    reduce = tuple((i for i in range(w.ndim) if i != axis))
    maximum = np.max(np.abs(w), axis=reduce, keepdims=True)
    scale = np.where(maximum > 0, maximum / 127, 1)
    return (np.clip(np.rint(w / scale), -127, 127).astype(np.int8), scale)
