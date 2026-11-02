def q267(x, weights, stride=2, padding=0):
    """Compute transposed convolution directly by scatter-add. X is NCHW; weights has shape
    (input_channels,output_channels,kh,kw). Use no output padding.
    """
    n, cin, h, w = x.shape
    _, cout, kh, kw = weights.shape
    oh = (h - 1) * stride - 2 * padding + kh
    ow = (w - 1) * stride - 2 * padding + kw
    out = x.new_zeros((n, cout, oh, ow))
    for i in range(h):
        for j in range(w):
            for a in range(kh):
                for b in range(kw):
                    y = i * stride - padding + a
                    z = j * stride - padding + b
                    if 0 <= y < oh and 0 <= z < ow:
                        out[:, :, y, z] += x[:, :, i, j] @ weights[:, :, a, b]
    return out
