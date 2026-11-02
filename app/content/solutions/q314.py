import numpy as np


def q314(query_length, key_length, query_offset=0, key_offset=0):
    """Build a causal permission mask using absolute query and key positions, supporting
    decoding against a longer cache.
    """
    # Absolute positions let a short query block attend correctly to a longer KV cache.
    return (np.arange(key_length) + key_offset)[None, :] <= (
        np.arange(query_length) + query_offset
    )[:, None]
