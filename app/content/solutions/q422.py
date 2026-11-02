import numpy as np


def q422(
    expert_ids,
    source_ranks,
    expert_to_rank,
    hidden_size,
    bytes_per_element=2,
):
    """Count dispatched token copies per expert and cross-rank dispatch-plus-return bytes.
    expert_ids is (tokens,top_k); each token's source rank is given. Ignore routing
    metadata and dropped tokens.
    """
    ids = np.asarray(expert_ids, int)
    owners = np.asarray(expert_to_rank, int)
    source = np.asarray(source_ranks, int)
    remote = owners[ids] != source[:, None]
    return (
        np.bincount(ids.ravel(), minlength=len(owners)),
        int(remote.sum() * hidden_size * bytes_per_element * 2),
    )
