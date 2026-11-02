import numpy as np


def q166(predictions, in_bag, labels):
    """Compute out-of-bag classification accuracy from per-tree predictions and a Boolean
    in_bag matrix of shape (trees,samples). Exclude samples that have no out-of-bag
    votes.
    """
    predictions, in_bag = (
        np.asarray(predictions),
        np.asarray(in_bag, bool),
    )
    labels = np.asarray(labels)
    correct = []
    for i in range(len(labels)):
        votes = predictions[~in_bag[:, i], i]
        if len(votes):
            classes, counts = np.unique(votes, return_counts=True)
            correct.append(classes[counts.argmax()] == labels[i])
    return float(np.mean(correct)) if correct else np.nan
