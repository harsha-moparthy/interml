import numpy as np


def binary_cross_entropy_with_logits(logits, targets):
    """Mean BCE for finite matching arrays; targets may be soft labels in [0,1].

    max(z,0) - z*y + log1p(exp(-abs(z))) avoids sigmoid saturation and exp(z).
    """
    z, y = np.asarray(logits, dtype=float), np.asarray(targets, dtype=float)
    if (
        z.shape != y.shape
        or not z.size
        or not np.isfinite(z).all()
        or not np.isfinite(y).all()
        or np.any((y < 0) | (y > 1))
    ):
        raise ValueError("finite matching inputs and targets in [0,1] required")
    return float((np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))).mean())
