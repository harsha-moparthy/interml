import numpy as np


def q414(x, block_size=64):
    """Quantize a flat array to the standard 16-level NF4 codebook using per-block absolute
    maxima. Return 4-bit indices stored in uint8, block scales, and reconstruction; the
    final block may be short.
    """
    levels = np.array(
        [
            -1,
            -0.6961928,
            -0.5250731,
            -0.3949175,
            -0.2844414,
            -0.1847734,
            -0.09105,
            0,
            0.0795803,
            0.1609302,
            0.2461123,
            0.3379152,
            0.4407098,
            0.562617,
            0.7229568,
            1,
        ]
    )
    x = np.asarray(x, float).ravel()
    codes = np.empty(len(x), np.uint8)
    rebuilt = np.empty_like(x)
    scales = []
    if block_size < 1:
        raise ValueError("positive block size required")
    for start in range(0, len(x), block_size):
        block = x[start : start + block_size]
        scale = np.abs(block).max()
        scale = scale if scale else 1.0
        index = np.argmin(np.abs(block[:, None] / scale - levels), axis=1)
        codes[start : start + len(block)] = index
        rebuilt[start : start + len(block)] = levels[index] * scale
        scales.append(scale)
    return (codes, np.array(scales), rebuilt)
