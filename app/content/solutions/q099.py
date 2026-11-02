import numpy as np


def q099(joint):
    """Compute mutual information in nats from a normalized joint probability matrix. Sum
    only positive joint-probability entries.
    """
    p = np.asarray(joint, float)
    independent = p.sum(1)[:, None] * p.sum(0)[None, :]
    keep = p > 0
    return float(np.sum(p[keep] * np.log(p[keep] / independent[keep])))
