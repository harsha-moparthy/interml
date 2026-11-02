import numpy as np


def q315(weights, probability, training=True, seed=0):
    """Apply inverted dropout to attention weights without renormalizing rows. Preserve
    expected weights during training; return a copy in evaluation mode.
    """
    if not 0 <= probability < 1:
        raise ValueError("invalid dropout probability")
    weights = np.asarray(weights, float)
    if not training:
        return weights.copy()
    # Do not renormalize surviving weights: inverted dropout preserves their expectation.
    keep = np.random.default_rng(seed).random(weights.shape) >= probability
    return weights * keep / (1 - probability)
