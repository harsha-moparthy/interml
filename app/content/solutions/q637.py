import numpy as np


def gaussian_mixture(x, components, *, iterations=100, seed=0, variance_floor=1e-6):
    """Diagonal-covariance GMM EM in pure NumPy; return weights, means, variances, r.

    A diagonal model assumes conditional feature independence within each component.
    Flooring variances prevents singular components; it is a modeling constraint.
    """
    x = np.asarray(x, dtype=float)
    if (
        x.ndim != 2
        or not 1 <= components <= len(x)
        or variance_floor <= 0
        or iterations < 1
        or not np.isfinite(x).all()
    ):
        raise ValueError("invalid GMM inputs")
    rng = np.random.default_rng(seed)
    means = x[rng.choice(len(x), components, replace=False)].copy()
    variances = np.tile(np.maximum(x.var(0), variance_floor), (components, 1))
    weights = np.full(components, 1 / components)

    def expectation():
        log_joint = np.log(weights) - 0.5 * (
            np.log(2 * np.pi * variances)[None]
            + (x[:, None] - means[None]) ** 2 / variances[None]
        ).sum(-1)
        unnormalized = np.exp(log_joint - log_joint.max(-1, keepdims=True))
        return unnormalized / unnormalized.sum(-1, keepdims=True)

    for _ in range(iterations):
        responsibilities = expectation()
        counts = responsibilities.sum(0)
        active = counts > np.finfo(float).eps
        means[active] = (responsibilities.T @ x)[active] / counts[active, None]
        squared = (x[:, None] - means[None]) ** 2
        variances[active] = (responsibilities[:, :, None] * squared).sum(0)[
            active
        ] / counts[active, None]
        variances = np.maximum(variances, variance_floor)
        weights = np.maximum(counts, np.finfo(float).eps)
        weights /= weights.sum()
    return weights, means, variances, expectation()
