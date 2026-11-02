import numpy as np


def q276(anchors, boxes=None, deltas=None):
    """Encode boxes relative to anchors as normalized center offsets and log size ratios,
    or decode supplied deltas. Provide exactly one of boxes/deltas; all boxes have
    positive sizes.
    """
    anchors = np.asarray(anchors, float)
    size = anchors[:, 2:] - anchors[:, :2]
    center = (anchors[:, 2:] + anchors[:, :2]) / 2
    if (boxes is None) == (deltas is None):
        raise ValueError("choose encode or decode")
    if boxes is not None:
        boxes = np.asarray(boxes, float)
        bs = boxes[:, 2:] - boxes[:, :2]
        bc = (boxes[:, 2:] + boxes[:, :2]) / 2
        return np.column_stack([(bc - center) / size, np.log(bs / size)])
    deltas = np.asarray(deltas, float)
    bc = center + deltas[:, :2] * size
    bs = size * np.exp(deltas[:, 2:])
    return np.column_stack([bc - bs / 2, bc + bs / 2])
