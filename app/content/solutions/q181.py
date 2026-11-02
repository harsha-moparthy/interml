import numpy as np
from scipy.special import logsumexp


def q181(x, k, iterations=100, seed=0):
    """Fit a diagonal-covariance GMM using EM. Initialize means from distinct rows, use
    empirical diagonal variance, and floor variances at 1e-6. Return weights, means,
    variances, and final responsibilities.
    """
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.choice(len(x), k, replace=False)].copy()
    variance = np.tile(np.maximum(x.var(0), 1e-06), (k, 1))
    weights = np.full(k, 1 / k)

    # Normalize responsibilities in log space to avoid underflow in Gaussian densities.
    def expectation():
        log_probabilities = np.log(weights)[None, :] - 0.5 * (
            np.log(2 * np.pi * variance)[None, :, :]
            + (x[:, None, :] - means[None, :, :]) ** 2 / variance[None, :, :]
        ).sum(-1)
        return np.exp(
            log_probabilities - logsumexp(log_probabilities, axis=1, keepdims=True)
        )

    for _ in range(iterations):
        responsibilities = expectation()
        counts = responsibilities.sum(0) + 1e-12
        weights = counts / counts.sum()
        means = responsibilities.T @ x / counts[:, None]
        variance = np.maximum(
            (
                responsibilities[:, :, None] * (x[:, None, :] - means[None, :, :]) ** 2
            ).sum(0)
            / counts[:, None],
            1e-06,
        )
    return (weights, means, variance, expectation())
