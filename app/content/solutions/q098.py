import numpy as np


def q098(p, q):
    """Compute symmetric Jensen-Shannon divergence in nats between two probability vectors.
    Handle zero probabilities by omitting zero-weight terms.
    """
    p, q = (np.asarray(p, float), np.asarray(q, float))
    m = (p + q) / 2

    def kl(a):
        keep = a > 0
        return np.sum(a[keep] * np.log(a[keep] / m[keep]))

    return float((kl(p) + kl(q)) / 2)
