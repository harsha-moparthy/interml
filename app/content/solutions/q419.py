import numpy as np


def q419(keys, values):
    """Quantize K and V independently to symmetric INT8 with one scale per last-axis
    vector. Return each tensor's (codes,scales,reconstruction).
    """
    result = []
    for tensor in (keys, values):
        x = np.asarray(tensor, float)
        maximum = np.max(np.abs(x), axis=-1, keepdims=True)
        scale = np.where(maximum > 0, maximum / 127, 1)
        codes = np.clip(np.rint(x / scale), -127, 127).astype(np.int8)
        result.append((codes, scale, codes.astype(float) * scale))
    return tuple(result)
