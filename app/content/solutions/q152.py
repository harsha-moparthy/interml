import numpy as np


def q152(x, y, c=1, epochs=200):
    """Solve the no-intercept linear SVM dual by coordinate ascent with 0<=alpha<=C. Return
    primal weights and dual variables. This variant intentionally has no equality
    constraint for a bias term.
    """
    x, y = (np.asarray(x, float), np.asarray(y, float))
    a = np.zeros(len(x))
    w = np.zeros(x.shape[1])
    for _ in range(epochs):
        for i in range(len(x)):
            curvature = x[i] @ x[i]
            if curvature == 0:
                continue
            updated = np.clip(a[i] + (1 - y[i] * (x[i] @ w)) / curvature, 0, c)
            w += (updated - a[i]) * y[i] * x[i]
            a[i] = updated
    return (w, a)
