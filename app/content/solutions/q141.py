import numpy as np


def q141(x, y, steps=50, ridge=1e-06):
    """Fit Poisson regression with a log link by iteratively reweighted least squares.
    Include an intercept; use a small ridge term for numerical stability.
    """
    x = np.column_stack([np.ones(len(x)), np.asarray(x, float)])
    y = np.asarray(y, float)
    w = np.zeros(x.shape[1])
    for _ in range(steps):
        eta = np.clip(x @ w, -20, 20)
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        w = np.linalg.solve(
            x.T @ (mu[:, None] * x) + ridge * np.eye(x.shape[1]),
            x.T @ (mu * z),
        )
    return w
