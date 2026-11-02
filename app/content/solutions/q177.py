import numpy as np


def q177(x, k):
    """Implement agglomerative single-link clustering by repeatedly merging the closest
    clusters until k remain. Return integer cluster labels; ties follow current cluster
    order.
    """
    x = np.asarray(x, float)
    distance = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    groups = [[i] for i in range(len(x))]
    if not 1 <= k <= len(x):
        raise ValueError("invalid k")
    while len(groups) > k:
        _, i, j = min(
            (
                (distance[np.ix_(a, b)].min(), i, j)
                for i, a in enumerate(groups)
                for j, b in enumerate(groups)
                if i < j
            )
        )
        groups[i] += groups[j]
        groups.pop(j)
    labels = np.empty(len(x), int)
    for i, group in enumerate(groups):
        labels[group] = i
    return labels
