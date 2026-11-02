import numpy as np


def q310(x, w_query, w_key, w_value):
    """Compute numerically stable single-head self-attention for X(length,width) using
    supplied projection matrices. Subtract row maxima before exponentiating; return
    context without rounding.
    """
    x = np.asarray(x, float)
    queries = x @ w_query
    keys = x @ w_key
    values = x @ w_value
    scores = queries @ keys.T / np.sqrt(queries.shape[-1])
    # Subtracting a row constant leaves softmax unchanged and prevents overflow.
    scores -= scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ values
