"""Return the fraction of predicted labels equal to their corresponding true labels."""

import numpy as np


def accuracy_score(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy
