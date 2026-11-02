import numpy as np


def stratified_kfold(labels, folds=5, *, seed=0):
    """Yield train/validation row indices; each class must contain at least folds rows.

    Shuffles within classes with a local RNG. Rotating the starting fold prevents
    every class remainder from overloading fold zero. Use group/time splits instead
    when examples are dependent; label balance alone does not prevent leakage.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or not 2 <= folds <= len(labels):
        raise ValueError("invalid labels or number of folds")
    rng = np.random.default_rng(seed)
    assignments = np.empty(len(labels), dtype=int)
    offset = 0
    for label in np.unique(labels):
        rows = np.flatnonzero(labels == label)
        if len(rows) < folds:
            raise ValueError("every class needs at least one example per fold")
        rng.shuffle(rows)
        assignments[rows] = (np.arange(len(rows)) + offset) % folds
        offset = (offset + len(rows)) % folds
    for fold in range(folds):
        yield np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)
