import math

import torch


def sinusoidal_encoding(
    length, model_dim, *, offset=0, device=None, dtype=torch.float32
):
    """Return (length, model_dim) positional encodings, including odd model widths.

    offset allows an incremental decoder to use absolute positions.
    No trainable parameters; time and storage are O(length * model_dim).
    """
    if length < 0 or model_dim <= 0 or offset < 0:
        raise ValueError("length/offset must be nonnegative and model_dim positive")
    work_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    positions = torch.arange(offset, offset + length, device=device, dtype=work_dtype)
    frequencies = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, model_dim, 2, device=device, dtype=work_dtype)
        / model_dim
    )
    angles = positions[:, None] * frequencies
    encoding = torch.empty(length, model_dim, device=device, dtype=work_dtype)
    encoding[:, 0::2] = angles.sin()
    encoding[:, 1::2] = angles[:, : model_dim // 2].cos()
    return encoding.to(dtype)
