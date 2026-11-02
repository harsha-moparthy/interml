import numpy as np


def q283(x, y, alpha=0.4, seed=0):
    """Apply MixUp to a batch and its one-hot targets, using one Beta(alpha,alpha)
    coefficient and a random batch permutation. Return mixed inputs, mixed targets, and
    coefficient.
    """
    rng = np.random.default_rng(seed)
    coefficient = float(rng.beta(alpha, alpha))
    permutation = rng.permutation(len(x))
    return (
        coefficient * x + (1 - coefficient) * x[permutation],
        coefficient * y + (1 - coefficient) * y[permutation],
        coefficient,
    )
