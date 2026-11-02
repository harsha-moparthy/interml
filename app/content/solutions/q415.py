import numpy as np


def q415(scales):
    """Double-quantize a vector of positive first-level scales: subtract its mean and
    symmetrically INT8-quantize the residuals. Return codes, residual scale, mean
    offset, and reconstructed scales.
    """
    x = np.asarray(scales, float)
    offset = x.mean()
    residual = x - offset
    maximum = np.abs(residual).max()
    scale = maximum / 127 if maximum else 1.0
    codes = np.clip(np.rint(residual / scale), -127, 127).astype(np.int8)
    return (
        codes,
        scale,
        float(offset),
        codes.astype(float) * scale + offset,
    )
