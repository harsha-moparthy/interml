import math


def q513(height, width, patch=16, decoder_scales=(2, 2), target_scale=4):
    """Return patch grid, decoder output size, target size, and missing spatial factor as
    exact rational numbers. The image dimensions must be patch-divisible. Example: 96/16
    -> 6; two 2x stages -> 24; target 384 requires another 16x. This is a shape planner,
    not an image reconstruction model.
    """
    from fractions import Fraction

    if (
        min(height, width, patch, target_scale) < 1
        or height % patch
        or width % patch
        or any((s < 1 for s in decoder_scales))
    ):
        raise ValueError("invalid dimensions")
    h, w = (height // patch, width // patch)
    factor = math.prod(decoder_scales)
    output = (h * factor, w * factor)
    target = (height * target_scale, width * target_scale)
    return (
        (h, w),
        output,
        target,
        (Fraction(target[0], output[0]), Fraction(target[1], output[1])),
    )
