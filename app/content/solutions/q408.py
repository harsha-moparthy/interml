import numpy as np


def q408(x):
    """Symmetrically quantize a finite array to signed INT8 using the range [-127,127].
    Return quantized values, scalar scale, and dequantized values; all-zero input uses
    scale 1.
    """
    x = np.asarray(x, float)
    maximum = np.max(np.abs(x))
    # All-zero input uses scale one to avoid dividing by zero.
    scale = maximum / 127 if maximum else 1.0
    q = np.clip(np.rint(x / scale), -127, 127).astype(np.int8)
    return (q, scale, q.astype(float) * scale)
