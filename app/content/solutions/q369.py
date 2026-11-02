import numpy as np


def q369(rewards):
    """Compute leave-one-out REINFORCE advantages within each row of rewards. Require at
    least two completions per group.
    """
    r = np.asarray(rewards, float)
    k = r.shape[-1]
    if k < 2:
        raise ValueError("group size must exceed one")
    return r - (r.sum(-1, keepdims=True) - r) / (k - 1)
