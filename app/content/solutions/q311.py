import numpy as np


def q311(query, keys, values, w_query, w_key, scoring_vector):
    """Compute Bahdanau additive attention for one query vector and a sequence of key/value
    vectors. Score each key with v dot tanh(query@Wq+key@Wk), normalize, and return
    context and weights.
    """
    scores = (
        np.tanh(np.asarray(query) @ w_query + np.asarray(keys) @ w_key) @ scoring_vector
    )
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    return (weights @ values, weights)
