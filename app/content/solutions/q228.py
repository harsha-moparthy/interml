import numpy as np
from scipy.special import logsumexp


def q228(logits, labels, smoothing=0.1):
    """Compute mean label-smoothed cross-entropy using target mass (1-epsilon) on the
    correct class plus epsilon/K on every class.
    """
    x = np.asarray(logits, float)
    logp = x - logsumexp(x, axis=1, keepdims=True)
    return float(
        np.mean(
            -(1 - smoothing) * logp[np.arange(len(x)), labels]
            - smoothing * logp.mean(1)
        )
    )
