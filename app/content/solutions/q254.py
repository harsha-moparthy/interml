import numpy as np


def q254(step, first_period, period_multiplier=2, peak=0.001, min_lr=0):
    """Compute cosine annealing with warm restarts. Each cycle grows by an integer
    period_multiplier>=1; the learning rate resets at cycle boundaries.
    """
    if first_period <= 0 or period_multiplier < 1 or step < 0:
        raise ValueError("invalid schedule")
    period = first_period
    while step >= period:
        step -= period
        period *= period_multiplier
    return float(min_lr + 0.5 * (peak - min_lr) * (1 + np.cos(np.pi * step / period)))
