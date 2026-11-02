import numpy as np


def q129(labels, folds=5, seed=0):
    """Create stratified k-fold index pairs by splitting each shuffled class across folds.
    Require at least folds examples of every class.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    held = [[] for _ in range(folds)]
    # Distribute each class independently so folds retain class balance.
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        if len(indices) < folds:
            raise ValueError("too few samples in a class")
        for i, part in enumerate(np.array_split(indices, folds)):
            held[i].extend(part)
    all_indices = np.arange(len(labels))
    return [(np.setdiff1d(all_indices, v), np.array(v, dtype=int)) for v in held]
