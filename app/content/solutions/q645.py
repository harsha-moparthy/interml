import numpy as np


def linear_svm(x, y, *, regularization=0.01, learning_rate=0.1, iterations=1000):
    """Minimize mean hinge loss + regularization*||w||²/2 by subgradient descent.

    Labels are -1/+1; intercept is unregularized. Feature scaling matters.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (
        x.ndim != 2
        or not len(x)
        or y.shape != (len(x),)
        or not np.isin(y, [-1, 1]).all()
        or regularization < 0
        or learning_rate <= 0
        or iterations < 1
    ):
        raise ValueError("invalid SVM inputs")
    weight, bias = np.zeros(x.shape[1]), 0.0
    for step in range(iterations):
        active = y * (x @ weight + bias) < 1
        gradient = regularization * weight - x[active].T @ y[active] / len(x)
        rate = learning_rate / np.sqrt(step + 1)
        weight -= rate * gradient
        bias += rate * y[active].sum() / len(x)
    return weight, bias
