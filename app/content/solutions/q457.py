import numpy as np


def q457(successes, failures, seed=0):
    """Select a Bernoulli bandit arm using independent Beta(1+successes,1+failures)
    posterior samples. Return chosen index and all samples.
    """
    draws = np.random.default_rng(seed).beta(
        np.asarray(successes) + 1, np.asarray(failures) + 1
    )
    return (int(draws.argmax()), draws)
