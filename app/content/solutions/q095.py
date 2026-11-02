import numpy as np


def q095(values, repetitions=2000, confidence=0.95, seed=0):
    """Estimate a percentile bootstrap confidence interval for the sample mean. Resample n
    observations with replacement for each replicate.
    """
    x = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = np.array(
        [rng.choice(x, len(x), replace=True).mean() for _ in range(repetitions)]
    )
    tail = (1 - confidence) / 2
    return np.quantile(means, [tail, 1 - tail])
