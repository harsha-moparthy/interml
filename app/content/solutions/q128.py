import numpy as np


def q128(labels, test_fraction=0.2, seed=0):
    """Return stratified train/test indices by shuffling each class separately and
    assigning floor(class_size*test_fraction) items to test.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train = []
    test = []
    for label in np.unique(labels):
        indices = rng.permutation(np.flatnonzero(labels == label))
        cut = int(len(indices) * test_fraction)
        test.extend(indices[:cut])
        train.extend(indices[cut:])
    return (np.array(train, dtype=int), np.array(test, dtype=int))
