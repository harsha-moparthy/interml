import numpy as np
from scipy.special import logsumexp


def q179(x, weights, means, covariances):
    """Compute GMM responsibilities with full positive-definite covariance matrices.
    Evaluate component log densities and normalize with log-sum-exp.
    """
    x = np.asarray(x, float)
    scores = []
    for weight, mean, cov in zip(weights, means, covariances):
        delta = x - mean
        _, logdet = np.linalg.slogdet(cov)
        quadratic = np.sum(delta * np.linalg.solve(cov, delta.T).T, axis=1)
        scores.append(
            np.log(weight) - 0.5 * (x.shape[1] * np.log(2 * np.pi) + logdet + quadratic)
        )
    scores = np.stack(scores, axis=1)
    return np.exp(scores - logsumexp(scores, axis=1, keepdims=True))
