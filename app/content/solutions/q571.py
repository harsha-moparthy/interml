import math


def q571(mechanisms, epsilon_limit, delta_limit):
    """Compose supplied valid (epsilon,delta) guarantees by BASIC SEQUENTIAL addition.
    Return totals, raising if negative/nonfinite components or configured limits are
    exceeded. Each repeated private selection/tuning/training release must appear
    separately; this ledger does not calculate DP-SGD epsilon from noise.
    """
    epsilon = delta = 0.0
    for e, d in mechanisms:
        if not math.isfinite(e + d) or e < 0 or (not 0 <= d < 1):
            raise ValueError("invalid guarantee")
        epsilon += e
        delta += d
    if epsilon > epsilon_limit or delta > delta_limit:
        raise ValueError("privacy budget exceeded")
    return (epsilon, delta)
