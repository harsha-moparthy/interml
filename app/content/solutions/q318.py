import numpy as np


def q318(query, key, value):
    """Implement unmasked multi-head attention with einsum. Q,K,V have shape (B,H,L,D),
    (B,H,S,D), (B,H,S,Dv).
    """
    queries, keys, values = (
        np.asarray(query, float),
        np.asarray(key, float),
        np.asarray(value, float),
    )
    # The einsum contracts head width and retains separate query/key positions.
    scores = np.einsum("bhld,bhsd->bhls", queries, keys) / np.sqrt(queries.shape[-1])
    scores -= scores.max(-1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(-1, keepdims=True)
    return np.einsum("bhls,bhsv->bhlv", weights, values)
