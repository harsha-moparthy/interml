def q285(images, patch):
    """Patchify NCHW images into (batch,number_of_patches,channels*patch^2), traversing
    patches row-major. Height and width must be divisible by patch.
    """
    n, c, h, w = images.shape
    if h % patch or w % patch:
        raise ValueError("incomplete patch")
    return (
        images.reshape(n, c, h // patch, patch, w // patch, patch)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(n, -1, c * patch * patch)
    )
