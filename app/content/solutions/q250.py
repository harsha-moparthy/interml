import numpy as np


def q250(weights, gradient, momentum, lr=0.01, beta=0.95, iterations=5):
    """Apply a Muon-style matrix update: momentum accumulation, norm scaling, and
    Newton-Schulz approximate orthogonalization. Return updated weights and momentum.
    This is an educational matrix-only variant.
    """
    m = beta * np.asarray(momentum) + (1 - beta) * gradient
    x = m.copy()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x /= np.linalg.norm(x) + 1e-07
    for _ in range(iterations):
        a = x @ x.T
        x = 3.4445 * x + (-4.775 * a + 2.0315 * a @ a) @ x
    if transposed:
        x = x.T
    return (weights - lr * x, m)
