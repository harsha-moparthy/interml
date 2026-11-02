import numpy as np


def q293(token_ids, pad_id=0):
    """Return a Boolean mask where token IDs differ from the padding ID. True means a real
    token.
    """
    return np.asarray(token_ids) != pad_id
