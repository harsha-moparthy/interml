import numpy as np


def q409(x, bits=8):
    """Asymmetrically quantize to unsigned integers using a representable range that
    includes zero. Return q, scale, integer zero point, and reconstruction; support 1
    through 16 bits.
    """
    if not 1 <= bits <= 16:
        raise ValueError("invalid bit width")
    x = np.asarray(x, float)
    low = min(float(x.min()), 0)
    high = max(float(x.max()), 0)
    top = 2**bits - 1
    scale = (high - low) / top if high > low else 1.0
    zero = int(np.clip(np.rint(-low / scale), 0, top))
    q = np.clip(np.rint(x / scale) + zero, 0, top).astype(np.uint16)
    return (q, scale, zero, (q.astype(float) - zero) * scale)
