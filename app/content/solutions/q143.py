import numpy as np


def q143(x, y, test, epsilon=1e-09):
    """Train Gaussian naive Bayes using independent feature variances, add epsilon to
    variances, and predict arbitrary class labels for test rows.
    """
    x, y, test = (
        np.asarray(x, float),
        np.asarray(y),
        np.asarray(test, float),
    )
    classes = np.unique(y)
    scores = []
    for label in classes:
        group = x[y == label]
        mean = group.mean(0)
        variance = group.var(0) + epsilon
        scores.append(
            np.log(len(group) / len(x))
            - 0.5
            * (np.log(2 * np.pi * variance) + (test - mean) ** 2 / variance).sum(1)
        )
    return classes[np.argmax(np.stack(scores, axis=1), axis=1)]
