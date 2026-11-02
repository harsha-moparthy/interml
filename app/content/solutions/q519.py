import math
import torch


def q519(height, width, window=4, shift=2):
    """Return allowed-attention masks [num_windows,M*M,M*M] after padding and a negative
    cyclic shift. Real tokens may attend only within the same nonwrapped shifted-window
    region; padded queries/keys are disallowed. Callers must handle all-masked padded
    query rows rather than softmaxing -inf blindly.
    """
    if min(height, width, window) < 1 or not 0 <= shift < window:
        raise ValueError("invalid grid")
    hp = math.ceil(height / window) * window
    wp = math.ceil(width / window) * window
    y, x = torch.meshgrid(torch.arange(hp), torch.arange(wp), indexing="ij")
    oy = (y + shift) % hp
    ox = (x + shift) % wp
    valid = (oy < height) & (ox < width)
    ry = (oy - shift) // window
    rx = (ox - shift) // window

    def partition(z):
        return (
            z.reshape(hp // window, window, wp // window, window)
            .permute(0, 2, 1, 3)
            .reshape(-1, window * window)
        )

    ry, rx, valid = map(partition, (ry, rx, valid))
    return (
        (ry[:, :, None] == ry[:, None, :])
        & (rx[:, :, None] == rx[:, None, :])
        & valid[:, :, None]
        & valid[:, None, :]
    )
