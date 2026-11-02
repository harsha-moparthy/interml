"""Build sinusoidal Transformer positional encodings from positions and model width."""

import numpy as np


def pos_encoding(position: int, d_model: int):
    if position == 0 or d_model <= 0:
        return -1
    pos = np.arange(position, dtype=np.float32).reshape(position, 1)
    ind = np.arange(d_model, dtype=np.float32).reshape(1, d_model)
    angle_rads = pos / np.power(10000, 2 * (ind // 2) / d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads.astype(np.float16)
