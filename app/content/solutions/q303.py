import numpy as np


def q303(token_ids, embedding_table):
    """Gather embedding rows for an arbitrarily shaped integer token-ID array. Append the
    embedding dimension to the output shape.
    """
    return np.asarray(embedding_table)[np.asarray(token_ids, dtype=int)]
