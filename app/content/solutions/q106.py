import numpy as np


def q106(logits, labels):
    """Compute the derivative of mean multiclass cross-entropy with respect to batched
    logits. Labels are integer class indices.
    """
    z = np.asarray(logits, float)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    p[np.arange(len(p)), labels] -= 1
    return p / len(p)
