import numpy as np


def q420(worker_gradients):
    """Simulate DDP gradient averaging: workers supply equal-length lists of same-shaped
    NumPy gradient arrays. Return the elementwise mean for every parameter; assume equal
    effective batch sizes.
    """
    if not worker_gradients:
        raise ValueError("workers required")
    return [
        np.mean(np.stack(group), axis=0)
        for group in zip(*worker_gradients, strict=True)
    ]
