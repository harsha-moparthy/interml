import numpy as np


def q253(step, warmup, total, peak, min_lr=0):
    """Compute linear warmup followed by cosine decay. At step zero return zero when warmup
    is positive; after total steps return min_lr. Require 0<=warmup<total.
    """
    if not 0 <= warmup < total:
        raise ValueError("invalid schedule")
    if step < warmup:
        return peak * max(step, 0) / warmup
    progress = np.clip((step - warmup) / (total - warmup), 0, 1)
    return float(min_lr + 0.5 * (peak - min_lr) * (1 + np.cos(np.pi * progress)))
