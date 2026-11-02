import numpy as np


def pairwise_iou(boxes, others):
    """Continuous-coordinate xyxy boxes; touching edges have zero intersection."""
    boxes, others = np.asarray(boxes, dtype=float), np.asarray(others, dtype=float)
    if (
        boxes.ndim != 2
        or others.ndim != 2
        or boxes.shape[1] != 4
        or others.shape[1] != 4
    ):
        raise ValueError("boxes must have shape (N,4)")
    if np.any(boxes[:, 2:] < boxes[:, :2]) or np.any(others[:, 2:] < others[:, :2]):
        raise ValueError("box maxima must not precede minima")
    overlap = np.maximum(
        np.minimum(boxes[:, None, 2:], others[None, :, 2:])
        - np.maximum(boxes[:, None, :2], others[None, :, :2]),
        0,
    )
    intersection = overlap.prod(-1)
    union = (
        (boxes[:, 2:] - boxes[:, :2]).prod(-1)[:, None]
        + (others[:, 2:] - others[:, :2]).prod(-1)[None, :]
        - intersection
    )
    return np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0
    )


def nms(boxes, scores, threshold=0.5):
    """Greedy single-class NMS; stable score ties preserve original index order.

    Compare the selected box against remaining boxes vectorially; selection itself
    is sequential. Worst-case O(N²) work and O(N) temporary storage.
    """
    boxes, scores = np.asarray(boxes), np.asarray(scores)
    if scores.shape != (len(boxes),) or not 0 <= threshold <= 1:
        raise ValueError("invalid scores or threshold")
    order = np.argsort(-scores, kind="stable")
    kept = []
    while len(order):
        current = order[0]
        kept.append(int(current))
        remaining = order[1:]
        overlap = pairwise_iou(boxes[current : current + 1], boxes[remaining])[0]
        order = remaining[overlap <= threshold]
    return np.asarray(kept, dtype=int)
