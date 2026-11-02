import numpy as np


def q374(token_log_probabilities, mask=None):
    """Compute perplexity from natural-log probabilities of observed tokens. An optional
    Boolean mask selects tokens; reject an empty selection.
    """
    x = np.asarray(token_log_probabilities, float)
    if mask is not None:
        x = x[np.asarray(mask, bool)]
    if not x.size:
        raise ValueError("no evaluated tokens")
    return float(np.exp(-x.mean()))
