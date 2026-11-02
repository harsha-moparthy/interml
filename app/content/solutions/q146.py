import numpy as np


def q146(x, y, test, ridge=1e-06):
    """Fit linear discriminant analysis with a shared within-class covariance matrix and
    empirical priors. Return class predictions.
    """
    x, y, test = (
        np.asarray(x, float),
        np.asarray(y),
        np.asarray(test, float),
    )
    classes = np.unique(y)
    means = np.stack([x[y == c].mean(0) for c in classes])
    covariance = np.zeros((x.shape[1], x.shape[1]))
    for c, mean in zip(classes, means):
        residual = x[y == c] - mean
        covariance += residual.T @ residual
    covariance /= len(x) - len(classes)
    covariance += ridge * np.eye(x.shape[1])
    solved = np.linalg.solve(covariance, means.T)
    prior = np.array([(y == c).mean() for c in classes])
    score = test @ solved - 0.5 * np.sum(means * solved.T, axis=1) + np.log(prior)
    return classes[score.argmax(1)]
