import numpy as np


def q082(values):
    """Compute count, mean, and sample variance from a stream using Welford's algorithm.
    Return NaN variance for fewer than two values.
    """
    count = 0
    mean = 0.0
    m2 = 0.0
    # Update the mean before the second deviation to avoid subtracting large squared sums.
    for x in values:
        count += 1
        delta = x - mean
        mean += delta / count
        m2 += delta * (x - mean)
    return (count, mean if count else np.nan, m2 / (count - 1) if count > 1 else np.nan)
