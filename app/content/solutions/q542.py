import numpy as np


def q542(scores, labels, k=10, sigma=1.0):
    """Return LambdaRank LOSS pseudo-gradients for one query using exponential gain
    2^label-1 and NDCG@k swap weights. Ranks are frozen for the step, equal-label pairs
    are skipped, and gradient descent subtracts these gradients. IDCG=0 or k=0 gives
    zeros. Ties follow input row order.
    """
    s = np.asarray(scores, float)
    y = np.asarray(labels, float)
    n = len(s)
    if k < 0 or sigma <= 0 or np.any(y < 0):
        raise ValueError("invalid ranking settings")
    order = np.argsort(-s, kind="stable")
    positions = np.empty(n, int)
    positions[order] = np.arange(n)
    discounts = np.where(np.arange(n) < k, 1 / np.log2(np.arange(n) + 2), 0)
    gain = np.exp2(y) - 1
    ideal = np.dot(np.sort(gain)[::-1], discounts)
    gradient = np.zeros(n)
    if ideal == 0:
        return gradient
    d = discounts[positions]
    for i in range(n):
        for j in range(n):
            if y[i] > y[j]:
                delta = abs((gain[i] - gain[j]) * (d[i] - d[j])) / ideal
                g = -sigma * np.exp(-np.logaddexp(0, sigma * (s[i] - s[j]))) * delta
                gradient[i] += g
                gradient[j] -= g
    return gradient
