import numpy as np
from scipy.special import logsumexp


def q230(anchors, positives, temperature=0.1):
    """Compute one-direction InfoNCE for paired embedding batches. L2-normalize vectors,
    use all positives as in-batch candidates, and place true matches on the diagonal.
    """
    a, b = (np.asarray(anchors, float), np.asarray(positives, float))
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    logits = a @ b.T / temperature
    return float(np.mean(logsumexp(logits, axis=1) - np.diag(logits)))
