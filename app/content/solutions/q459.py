import numpy as np


def q459(td_errors, batch_size, alpha=0.6, beta=0.4, epsilon=1e-06, seed=0):
    """Sample proportional prioritized replay indices with replacement and return
    normalized importance weights. Use priorities (abs(error)+epsilon)^alpha; normalize
    weights by the maximum over the entire buffer.
    """
    priority = (np.abs(td_errors) + epsilon) ** alpha
    p = priority / priority.sum()
    n = len(p)
    indices = np.random.default_rng(seed).choice(n, size=batch_size, p=p)
    weights = (n * p[indices]) ** (-beta)
    weights /= (n * p.min()) ** (-beta)
    return (indices, weights, p)
