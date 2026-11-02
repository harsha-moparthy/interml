import numpy as np


def q278(boxes, scores, sigma=0.5, min_score=0.001):
    """Implement Gaussian soft-NMS: repeatedly select the highest remaining score and decay
    other scores by exp(-IoU^2/sigma). Return selected original indices and their scores
    at selection time.
    """
    b = np.asarray(boxes, float)
    scores = np.array(scores, float, copy=True)
    remaining = list(range(len(b)))
    kept = []
    area = np.maximum(b[:, 2:] - b[:, :2], 0).prod(1)
    while remaining:
        i = max(remaining, key=lambda j: scores[j])
        remaining.remove(i)
        if scores[i] < min_score:
            break
        kept.append((i, float(scores[i])))
        for j in remaining:
            intersection = np.maximum(
                np.minimum(b[i, 2:], b[j, 2:]) - np.maximum(b[i, :2], b[j, :2]),
                0,
            ).prod()
            union = area[i] + area[j] - intersection
            iou = intersection / union if union else 0
            scores[j] *= np.exp(-iou * iou / sigma)
    return kept
