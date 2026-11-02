import numpy as np


def q192(log_likelihood, parameters, samples):
    """Compute Akaike and Bayesian information criteria from total log likelihood,
    parameter count, and positive sample count.
    """
    return (
        2 * parameters - 2 * log_likelihood,
        parameters * np.log(samples) - 2 * log_likelihood,
    )
