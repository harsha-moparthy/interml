import torch


def q518(tiles, origins, output_shape):
    """Blend CHW tiles at supplied nonnegative (top,left) OUTPUT coordinates using uniform
    overlap weights. Return the reconstructed CHW image. Reject uncovered pixels or
    out-of-bounds tiles. The caller handles model scaling/halo cropping before this
    stitching core.
    """
    if not tiles or len(tiles) != len(origins):
        raise ValueError("tiles/origins mismatch")
    output = tiles[0].new_zeros(output_shape)
    count = tiles[0].new_zeros(output_shape[-2:])
    for tile, (y, x) in zip(tiles, origins):
        c, h, w = tile.shape
        if (
            c != output_shape[0]
            or min(y, x) < 0
            or y + h > output_shape[1]
            or (x + w > output_shape[2])
        ):
            raise ValueError("tile outside output")
        output[:, y : y + h, x : x + w] += tile
        count[y : y + h, x : x + w] += 1
    if torch.any(count == 0):
        raise ValueError("uncovered output")
    return output / count
