def q336(x, attention, feed_forward, norm1, norm2, pre_norm=True):
    """Assemble either a pre-norm or post-norm transformer block from supplied callable
    sublayers. All sublayers preserve shape; dropout is omitted.
    """
    if pre_norm:
        x = x + attention(norm1(x))
        return x + feed_forward(norm2(x))
    x = norm1(x + attention(x))
    return norm2(x + feed_forward(x))
