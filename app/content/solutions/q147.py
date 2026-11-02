import numpy as np


def q147(x, y, test, ridge=1e-06):
    """Fit quadratic discriminant analysis with class-specific covariance matrices and
    empirical priors. Each class must have at least two samples.
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
        covariance = np.atleast_2d(np.cov(group, rowvar=False)) + ridge * np.eye(
            x.shape[1]
        )
        delta = test - group.mean(0)
        _, logdet = np.linalg.slogdet(covariance)
        score = -0.5 * (
            np.sum(delta * np.linalg.solve(covariance, delta.T).T, axis=1) + logdet
        )
        scores.append(score + np.log(len(group) / len(x)))
    return classes[np.stack(scores, axis=1).argmax(1)]
