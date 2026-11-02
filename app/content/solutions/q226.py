import numpy as np
from scipy.special import logsumexp


def q226(logits, labels):
    """Compute mean multiclass cross-entropy from a matrix of logits and integer class
    indices using log-sum-exp.
    """
    x = np.asarray(logits, float)
    return float(np.mean(logsumexp(x, axis=1) - x[np.arange(len(x)), labels]))
