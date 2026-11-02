import numpy as np


def q097(p, q):
    """Return entropy H(p) and cross-entropy H(p,q) in nats for probability vectors. Ignore
    terms with p=0; return infinite cross-entropy when q=0 where p>0.
    """
    p, q = (np.asarray(p, float), np.asarray(q, float))
    mask = p > 0
    entropy = -np.sum(p[mask] * np.log(p[mask]))
    cross = np.inf if np.any(q[mask] == 0) else -np.sum(p[mask] * np.log(q[mask]))
    return (float(entropy), float(cross))
