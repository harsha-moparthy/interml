import numpy as np


def q255(gradients, max_norm):
    """Clip a list of NumPy gradients by their combined L2 norm. Return independent clipped
    arrays and the original norm.
    """
    norm = np.sqrt(sum((np.sum(np.asarray(g, float) ** 2) for g in gradients)))
    factor = min(1, max_norm / (norm + 1e-12))
    return ([np.asarray(g) * factor for g in gradients], float(norm))
