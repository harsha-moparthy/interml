import numpy as np
from scipy.special import logsumexp


def q353(logits, targets, mask):
    """Sum selected token log probabilities per sequence. Logits are (B,L,V); targets and
    Boolean mask are (B,L). Targets must be valid vocabulary indices even in masked
    positions.
    """
    x = np.asarray(logits, float)
    logp = x - logsumexp(x, axis=-1, keepdims=True)
    chosen = np.take_along_axis(logp, np.asarray(targets)[..., None], axis=-1)[..., 0]
    return (chosen * np.asarray(mask, bool)).sum(-1)
