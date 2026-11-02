import numpy as np


def q208(confidence, correct, bins=10):
    """Compute expected calibration error from confidence values in [0,1] and Boolean
    correctness. Use equal-width bins with confidence one included in the last bin.
    """
    confidence = np.asarray(confidence, float)
    correct = np.asarray(correct, float)
    index = np.minimum((confidence * bins).astype(int), bins - 1)
    result = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            result += mask.mean() * abs(confidence[mask].mean() - correct[mask].mean())
    return float(result)
