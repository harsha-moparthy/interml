def q270(x):
    """Globally average the spatial dimensions of an NCHW tensor, returning shape
    (batch,channels).
    """
    return x.mean(dim=(-2, -1))
