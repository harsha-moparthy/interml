import numpy as np


def q188(x, test, trees=100, sample_size=128, seed=0):
    """Implement isolation-forest anomaly scoring with random feature thresholds and
    average path lengths. Return scores where larger means more anomalous. Require at
    least two training rows.
    """
    x, test = (np.asarray(x, float), np.asarray(test, float))
    rng = np.random.default_rng(seed)
    size = min(sample_size, len(x))
    if size < 2:
        raise ValueError("too few training rows")

    def correction(n):
        return (
            0.0 if n <= 1 else 2 * sum((1 / i for i in range(1, n))) - 2 * (n - 1) / n
        )

    def build(a, depth):
        usable = np.flatnonzero(np.ptp(a, axis=0) > 0)
        if len(a) <= 1 or depth == 0 or (not len(usable)):
            return len(a)
        feature = int(rng.choice(usable))
        threshold = rng.uniform(a[:, feature].min(), a[:, feature].max())
        mask = a[:, feature] < threshold
        return (
            feature,
            threshold,
            build(a[mask], depth - 1),
            build(a[~mask], depth - 1),
        )

    def path(tree, row, depth=0):
        if isinstance(tree, int):
            return depth + correction(tree)
        feature, threshold, left, right = tree
        return path(left if row[feature] < threshold else right, row, depth + 1)

    depths = np.zeros(len(test))
    for _ in range(trees):
        tree = build(
            x[rng.choice(len(x), size, replace=False)],
            int(np.ceil(np.log2(size))),
        )
        depths += np.array([path(tree, row) for row in test])
    return 2 ** (-depths / (trees * correction(size)))
