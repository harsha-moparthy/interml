import numpy as np


def q211(reference_counts, current_counts, epsilon=1e-06):
    """Compute population stability index over a common set of bins. Add epsilon to each
    count before normalization to handle empty bins.
    """
    p = np.asarray(reference_counts, float) + epsilon
    q = np.asarray(current_counts, float) + epsilon
    p /= p.sum()
    q /= q.sum()
    return float(np.sum((q - p) * np.log(q / p)))
