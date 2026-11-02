import numpy as np


def q175(x, centers, batch_size=16, steps=100, seed=0):
    """Update given k-means centers from random mini-batches using a per-center running
    mean. Return centers and nearest-center labels for every training row.
    """
    x = np.asarray(x, float)
    c = np.array(centers, float, copy=True)
    count = np.zeros(len(c))
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        for point in x[rng.integers(len(x), size=batch_size)]:
            j = np.argmin(((c - point) ** 2).sum(1))
            count[j] += 1
            c[j] += (point - c[j]) / count[j]
    labels = ((x[:, None] - c[None, :]) ** 2).sum(-1).argmin(1)
    return (c, labels)
