import numpy as np


def q537(query_ids, *arrays):
    """Stable-sort query IDs and aligned arrays into contiguous ranking groups. Return
    (sorted_arrays,group_sizes,permutation,inverse_permutation); query IDs are the first
    sorted array. No train/test split is performed by this grouping core.
    """
    ids = np.asarray(query_ids)
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    boundaries = (
        np.r_[
            0,
            1 + np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]),
            len(ids),
        ]
        if len(ids)
        else np.array([0])
    )
    if any((len(a) != len(ids) for a in arrays)):
        raise ValueError("row mismatch")
    return (
        [sorted_ids] + [np.asarray(a)[order] for a in arrays],
        np.diff(boundaries),
        order,
        np.argsort(order),
    )
