import numpy as np


def q566(population, rate, rng):
    """Return independently Poisson-sampled record IDs using a supplied NumPy Generator.
    Each of N records is included with probability q; empty batches are allowed, and
    fixed-size sampling is not substituted.
    """
    if population < 0 or not 0 <= rate <= 1:
        raise ValueError("invalid population/rate")
    return np.flatnonzero(rng.random(population) < rate)
