import numpy as np


def q308(document_ids):
    """Create a causal attention permission mask for packed documents. A token may attend
    to itself or earlier tokens from the same document only. IDs uniquely identify
    packed documents.
    """
    ids = np.asarray(document_ids)
    n = len(ids)
    return (ids[:, None] == ids[None, :]) & (
        np.arange(n)[:, None] >= np.arange(n)[None, :]
    )
