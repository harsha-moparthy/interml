import numpy as np


def q201(truth, prediction):
    """Compute Hamming loss: the fraction of incorrect entries in equally shaped, nonempty
    multilabel indicator arrays.
    """
    return float(np.mean(np.asarray(truth) != np.asarray(prediction)))
