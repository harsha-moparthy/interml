import numpy as np


def q277(boxes, scores, threshold=0.5):
    """Implement greedy nonmaximum suppression for xyxy boxes. Return original indices in
    descending selected-score order, with stable tie handling.
    """
    b = np.asarray(boxes, float)
    order = np.argsort(-np.asarray(scores), kind="stable")
    kept = []
    area = np.maximum(b[:, 2:] - b[:, :2], 0).prod(1)
    while len(order):
        i = int(order[0])
        kept.append(i)
        rest = order[1:]
        intersection = np.maximum(
            np.minimum(b[i, 2:], b[rest, 2:]) - np.maximum(b[i, :2], b[rest, :2]),
            0,
        ).prod(1)
        union = area[i] + area[rest] - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )
        order = rest[iou <= threshold]
    return kept
