def q510(tokens, height, width, registers=0):
    """Convert [B,1+R+H*W,C] ViT tokens to BCHW after removing CLS and R register tokens.
    The rectangular patch-grid dimensions are supplied explicitly; reject mismatched
    token counts.
    """
    b, n, c = tokens.shape
    if min(height, width) < 1 or registers < 0 or n != 1 + registers + height * width:
        raise ValueError("grid/prefix mismatch")
    return tokens[:, 1 + registers :].transpose(1, 2).reshape(b, c, height, width)
