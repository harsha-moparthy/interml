"""Return the k nearest points to a query under Euclidean distance."""

import numpy as np


def k_nearest_neighbors(points, query_point, k):
    if not points or k <= 0:
        return []
    if k > len(points):
        k = len(points)
    points_array = np.array(points)
    query_array = np.array(query_point)
    distances = np.sqrt(np.sum((points_array - query_array) ** 2, axis=1))
    k_nearest_indices = np.argsort(distances)[:k]
    return [tuple(points_array[i]) for i in k_nearest_indices]
