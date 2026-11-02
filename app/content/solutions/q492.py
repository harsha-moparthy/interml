import numpy as np


def q492(features, adjacency, weights):
    """Implement one GCN layer: ReLU(D^-1/2*(A+I)*D^-1/2*X*W) for an undirected nonnegative
    adjacency matrix, adding one self-loop at every node.
    """
    a = np.asarray(adjacency, float) + np.eye(len(adjacency))
    inverse = 1 / np.sqrt(a.sum(1))
    normalized = inverse[:, None] * a * inverse[None, :]
    return np.maximum(0, normalized @ features @ weights)
