def q286(x, window):
    """Partition an NHWC tensor into nonoverlapping square windows and return windows plus
    a reconstructed tensor demonstrating the inverse permutation. Spatial dimensions
    must be divisible by window.
    """
    n, h, w, c = x.shape
    tiles = (
        x.reshape(n, h // window, window, w // window, window, c)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(-1, window, window, c)
    )
    restored = (
        tiles.reshape(n, h // window, w // window, window, window, c)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(n, h, w, c)
    )
    return (tiles, restored)
