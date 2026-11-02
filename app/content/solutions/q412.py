import numpy as np


def q412(x, weights):
    """Quantize a (K,N) weight matrix per output column to signed INT4 [-7,7], pack two
    signed nibbles per byte, unpack, and compute X@dequantized_W. Return packed bytes,
    column scales, and output; pad an odd flat length.
    """
    w = np.asarray(weights, float)
    maximum = np.abs(w).max(axis=0)
    scale = np.where(maximum > 0, maximum / 7, 1)
    q = np.clip(np.rint(w / scale), -7, 7).astype(np.int8)
    flat = (q.ravel().astype(np.int16) & 15).astype(np.uint8)
    if flat.size % 2:
        flat = np.append(flat, np.uint8(0))
    packed = flat[::2] | flat[1::2] << 4
    unpack = np.empty(2 * len(packed), np.int16)
    unpack[::2] = packed & 15
    unpack[1::2] = packed >> 4
    unpack = np.where(unpack >= 8, unpack - 16, unpack)
    reconstructed = unpack[: w.size].reshape(w.shape) * scale
    return (packed, scale, np.asarray(x) @ reconstructed)
