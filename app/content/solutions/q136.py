import numpy as np


def q136(x, y, regularization=1):
    """Solve ridge regression with an intercept. Do not regularize the intercept. Return
    coefficients including the intercept first.
    """
    x = np.asarray(x, float)
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * regularization
    penalty[0, 0] = 0
    return np.linalg.solve(design.T @ design + penalty, design.T @ np.asarray(y))
