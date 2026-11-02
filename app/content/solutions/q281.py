import numpy as np


def q281(boxes, targets):
    """Compute mean generalized-IoU loss for paired positive-area xyxy boxes. Use the
    smallest enclosing rectangle to penalize nonoverlap.
    """
    a, b = (np.asarray(boxes, float), np.asarray(targets, float))
    intersection = np.maximum(
        np.minimum(a[:, 2:], b[:, 2:]) - np.maximum(a[:, :2], b[:, :2]), 0
    ).prod(1)
    union = (a[:, 2:] - a[:, :2]).prod(1) + (b[:, 2:] - b[:, :2]).prod(1) - intersection
    enclosing = (np.maximum(a[:, 2:], b[:, 2:]) - np.minimum(a[:, :2], b[:, :2])).prod(
        1
    )
    return float(np.mean(1 - intersection / union + (enclosing - union) / enclosing))
