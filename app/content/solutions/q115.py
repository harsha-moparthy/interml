import numpy as np


def q115(x, y, hidden=4, steps=500, lr=0.05, seed=0):
    """Train a tiny tanh MLP for scalar regression using manual reverse-mode derivatives
    and full-batch SGD. Return learned parameters and predictions. No automatic
    differentiation is allowed.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    y = np.asarray(y, float).reshape(-1, 1)
    w1 = rng.normal(0, 0.2, (x.shape[1], hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, 0.2, (hidden, 1))
    b2 = np.zeros(1)
    for _ in range(steps):
        h = np.tanh(x @ w1 + b1)
        prediction = h @ w2 + b2
        dy = 2 * (prediction - y) / len(x)
        dh = dy @ w2.T * (1 - h * h)
        dw2 = h.T @ dy
        dw1 = x.T @ dh
        w2 -= lr * dw2
        b2 -= lr * dy.sum(0)
        w1 -= lr * dw1
        b1 -= lr * dh.sum(0)
    return ((w1, b1, w2, b2), np.tanh(x @ w1 + b1) @ w2 + b2)
