import numpy as np


def q225(logits, targets):
    """Compute mean binary cross-entropy directly from logits without forming unstable
    probabilities. Targets lie in [0,1].
    """
    x, y = (np.asarray(logits, float), np.asarray(targets, float))
    return float(np.mean(np.maximum(x, 0) - x * y + np.log1p(np.exp(-np.abs(x)))))
