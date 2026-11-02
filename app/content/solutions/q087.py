import numpy as np


def q087(prior, likelihood):
    """Apply Bayes' rule to nonnegative prior and likelihood vectors; normalize their
    elementwise product into a posterior.
    """
    weights = np.asarray(prior, float) * likelihood
    if weights.sum() <= 0:
        raise ValueError("zero evidence")
    return weights / weights.sum()
