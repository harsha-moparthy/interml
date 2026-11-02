import numpy as np


def knn_predict(train, targets, query, k=5, *, classification=False):
    """Vectorized brute-force KNN; smallest sorted label wins vote ties.

    Distance expansion avoids an (queries,train,features) temporary, but the
    (queries,train) distance matrix still costs O(Q*N) memory. Chunk large queries.
    """
    train, query = np.asarray(train, dtype=float), np.asarray(query, dtype=float)
    targets = np.asarray(targets)
    if (
        train.ndim != 2
        or query.ndim != 2
        or train.shape[1] != query.shape[1]
        or len(targets) != len(train)
        or not 1 <= k <= len(train)
    ):
        raise ValueError("incompatible data or invalid k")
    squared = np.maximum(
        (query * query).sum(1)[:, None]
        + (train * train).sum(1)[None, :]
        - 2 * query @ train.T,
        0,
    )
    neighbors = np.argsort(squared, axis=1, kind="stable")[:, :k]
    if not classification:
        return targets[neighbors].mean(axis=1)
    classes, encoded = np.unique(targets, return_inverse=True)
    votes = (encoded[neighbors][..., None] == np.arange(len(classes))).sum(axis=1)
    return classes[votes.argmax(axis=1)]
