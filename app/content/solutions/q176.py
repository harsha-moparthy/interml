import numpy as np


def q176(x, epsilon, min_samples):
    """Implement DBSCAN with Euclidean neighborhoods including each point itself. Return
    cluster labels starting at zero, with -1 for noise. Previously marked noise can
    become a border point.
    """
    x = np.asarray(x, float)
    distance = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
    neighbors = [np.flatnonzero(row <= epsilon) for row in distance]
    labels = np.full(len(x), -1)
    visited = np.zeros(len(x), bool)
    cluster = 0
    for i in range(len(x)):
        if visited[i]:
            continue
        visited[i] = True
        if len(neighbors[i]) < min_samples:
            continue
        labels[i] = cluster
        queue = list(neighbors[i])
        queued = set(queue)
        for j in queue:
            if not visited[j]:
                visited[j] = True
                if len(neighbors[j]) >= min_samples:
                    for v in neighbors[j]:
                        if v not in queued:
                            queue.append(v)
                            queued.add(v)
            if labels[j] < 0:
                labels[j] = cluster
        cluster += 1
    return labels
