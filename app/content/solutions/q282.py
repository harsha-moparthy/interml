import numpy as np


def q282(truth, prediction, classes):
    """Compute mean segmentation IoU over classes with nonempty union. Ignore classes
    absent from both truth and prediction.
    """
    y, p = (np.asarray(truth), np.asarray(prediction))
    values = []
    for c in range(classes):
        union = np.sum((y == c) | (p == c))
        if union:
            values.append(np.sum((y == c) & (p == c)) / union)
    return float(np.mean(values)) if values else 0.0
