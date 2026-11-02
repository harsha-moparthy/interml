import numpy as np


def q227(prediction, target, delta=1):
    """Compute mean Huber loss: half squared error within delta and linear growth outside
    it. Require positive delta.
    """
    if delta <= 0:
        raise ValueError("invalid delta")
    error = np.abs(np.asarray(prediction) - target)
    return float(
        np.mean(
            np.where(
                error <= delta,
                0.5 * error**2,
                delta * (error - 0.5 * delta),
            )
        )
    )
