import numpy as np
from scipy.special import expit


def q300(center, contexts, labels, lr=0.01):
    """Perform one skip-gram negative-sampling SGD step for a center embedding and selected
    context embeddings. Labels indicate positive (1) or negative (0) pairs. Return
    simultaneous parameter updates and summed logistic loss.
    """
    center = np.asarray(center, float)
    contexts = np.asarray(contexts, float)
    labels = np.asarray(labels, float)
    scores = contexts @ center
    error = expit(scores) - labels
    loss = np.sum(np.logaddexp(0, scores) - labels * scores)
    return (
        center - lr * (error @ contexts),
        contexts - lr * error[:, None] * center[None, :],
        float(loss),
    )
