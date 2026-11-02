import numpy as np


def linear_regression(
    x,
    y,
    *,
    ridge=0.0,
    method="closed",
    epochs=1000,
    learning_rate=0.01,
    batch_size=32,
    seed=0
):
    """Fit squared loss plus ridge*||w||²; return intercept and coefficients.

    The objective is mean squared error / 2 + ridge*||w||² / 2.
    The intercept is unpenalized. Closed form uses a solve/lstsq, never an inverse.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 2 or y.shape != (len(x),) or not len(x) or ridge < 0 or batch_size < 1:
        raise ValueError("invalid regression inputs")
    design = np.column_stack((np.ones(len(x)), x))
    penalty = np.r_[0.0, np.ones(x.shape[1])]
    if method == "closed":
        if ridge == 0:
            parameters = np.linalg.lstsq(design, y, rcond=None)[0]
        else:
            parameters = np.linalg.solve(
                design.T @ design + len(x) * ridge * np.diag(penalty), design.T @ y
            )
    elif method == "sgd":
        if learning_rate <= 0 or epochs < 1:
            raise ValueError("positive learning rate and epoch count required")
        parameters = np.zeros(design.shape[1])
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(x), batch_size):
                ids = order[start : start + batch_size]
                gradient = (
                    design[ids].T @ (design[ids] @ parameters - y[ids]) / len(ids)
                )
                parameters -= learning_rate * (gradient + ridge * penalty * parameters)
    else:
        raise ValueError("method must be closed or sgd")
    return parameters[0], parameters[1:]
