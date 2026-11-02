import numpy as np


def q145(x, y, test, alpha=1):
    """Fit multinomial naive Bayes to nonnegative count vectors with additive smoothing.
    Return predicted class labels for test vectors.
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
        counts = group.sum(0) + alpha
        logp = np.log(counts / counts.sum())
        scores.append(test @ logp + np.log(len(group) / len(x)))
    return classes[np.argmax(np.stack(scores, axis=1), axis=1)]
