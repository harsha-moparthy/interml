import numpy as np


def q200(truth, prediction):
    """Compute binary Matthews correlation coefficient, returning zero when the denominator
    is zero.
    """
    y, p = (np.asarray(truth, bool), np.asarray(prediction, bool))
    tp = float(np.sum(y & p))
    tn = float(np.sum(~y & ~p))
    fp = float(np.sum(~y & p))
    fn = float(np.sum(y & ~p))
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator else 0.0
