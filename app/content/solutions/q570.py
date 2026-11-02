import numpy as np


def q570(utilities, epsilon, sensitivity, rng):
    """Sample an exponential-mechanism candidate proportional to
    exp(epsilon*u/(2*sensitivity)). Return sampled index and probabilities. Utilities
    must be finite/nonempty, epsilon>=0 and supplied sensitivity>0. The bound is an
    external assumption; accepting it does not prove privacy of utility construction.
    """
    u = np.asarray(utilities, float)
    if not u.size or not np.isfinite(u).all() or epsilon < 0 or (sensitivity <= 0):
        raise ValueError("invalid mechanism inputs")
    logits = (u - u.max()) * (epsilon / (2 * sensitivity))
    weights = np.exp(logits)
    weights /= weights.sum()
    return (int(rng.choice(len(u), p=weights)), weights)
