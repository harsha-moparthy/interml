import numpy as np


def q274(a, b):
    """Compute pairwise IoU for xyxy boxes, with continuous coordinates and no
    inclusive-pixel +1. Degenerate unions produce zero.
    """
    a, b = (np.asarray(a, float), np.asarray(b, float))
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    # Intersection widths/heights are clipped at zero for disjoint boxes.
    intersection = np.maximum(high - low, 0).prod(-1)
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(-1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(-1)
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )
