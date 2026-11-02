def q511(x, scale=2, inverse=False):
    """Implement PixelShuffle, or its inverse PixelUnshuffle, by reshape/permutation only.
    Input is BCHW; reject invalid divisibility. Forward maps [B,C*r*r,H,W] to
    [B,C,H*r,W*r].
    """
    b, c, h, w = x.shape
    r = scale
    if not isinstance(r, int) or r < 1:
        raise ValueError("positive integer scale required")
    if inverse:
        if h % r or w % r:
            raise ValueError("spatial divisibility")
        return (
            x.reshape(b, c, h // r, r, w // r, r)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(b, c * r * r, h // r, w // r)
        )
    if c % (r * r):
        raise ValueError("channel divisibility")
    return (
        x.reshape(b, c // (r * r), r, r, h, w)
        .permute(0, 1, 4, 2, 5, 3)
        .reshape(b, c // (r * r), h * r, w * r)
    )
