import numpy as np


def q090(values):
    """Estimate a Gaussian's mean and variance by maximum likelihood from a nonempty
    sample. Use denominator n, not n-1.
    """
    x = np.asarray(values, float)
    return (float(x.mean()), float(((x - x.mean()) ** 2).mean()))
