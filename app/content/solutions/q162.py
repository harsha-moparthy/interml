import numpy as np


def q162(x, y, max_depth=4):
    """Build a binary CART classifier with exhaustive Gini splits. Return nested
    dictionaries: leaves contain prediction; internal nodes contain feature, threshold,
    left, right, and prediction.
    """
    x, y = (np.asarray(x, float), np.asarray(y))

    def build(a, b, depth):
        classes, counts = np.unique(b, return_counts=True)
        node = {"prediction": classes[counts.argmax()]}
        if depth == 0 or len(classes) == 1:
            return node
        best = None
        best_loss = np.inf
        for j in range(a.shape[1]):
            values = np.unique(a[:, j])
            for threshold in (values[:-1] + values[1:]) / 2:
                mask = a[:, j] <= threshold
                score = 0
                for child in (b[mask], b[~mask]):
                    p = np.unique(child, return_counts=True)[1] / len(child)
                    score += len(child) * (1 - p @ p)
                if score < best_loss:
                    best_loss = score
                    best = (j, threshold, mask)
        if best is None:
            return node
        j, t, mask = best
        node.update(
            feature=j,
            threshold=float(t),
            left=build(a[mask], b[mask], depth - 1),
            right=build(a[~mask], b[~mask], depth - 1),
        )
        return node

    return build(x, y, max_depth)
