import numpy as np
from scipy.special import logsumexp


def q385(logits, temperature=1, seed=0):
    """Sample one token from finite one-dimensional logits divided by positive temperature;
    temperature zero selects argmax.
    """
    x = np.asarray(logits, float)
    if temperature < 0:
        raise ValueError("negative temperature")
    if temperature == 0:
        return int(x.argmax())
    x = x / temperature
    p = np.exp(x - logsumexp(x))
    return int(np.random.default_rng(seed).choice(len(p), p=p))
