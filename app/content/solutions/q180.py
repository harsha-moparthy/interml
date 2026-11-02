import numpy as np


def q180(x, responsibilities, epsilon=1e-06):
    """Perform a full-covariance GMM M-step. Every component must have positive effective
    count. Add epsilon to covariance diagonals; return weights, means, and covariances.
    """
    x, r = (np.asarray(x, float), np.asarray(responsibilities, float))
    count = r.sum(0)
    if np.any(count <= 0):
        raise ValueError("empty component")
    means = r.T @ x / count[:, None]
    covariance = []
    for j, mean in enumerate(means):
        delta = x - mean
        covariance.append(
            delta.T @ (r[:, j, None] * delta) / count[j] + epsilon * np.eye(x.shape[1])
        )
    return (count / len(x), means, np.stack(covariance))
