import numpy as np


def q500(prediction_times, feature_available_times):
    """Detect look-ahead leakage from a matrix of actual feature availability timestamps
    and one prediction timestamp per row. Return the violating cell mask and row
    indices; availability equal to prediction time is allowed.
    """
    mask = np.asarray(feature_available_times) > np.asarray(prediction_times)[:, None]
    return (mask, np.flatnonzero(mask.any(axis=1)))
