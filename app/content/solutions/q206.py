import numpy as np


def q206(fpr, tpr):
    """Integrate a supplied ROC curve by the trapezoidal rule. Require nondecreasing FPR
    and equally shaped arrays.
    """
    x, y = (np.asarray(fpr, float), np.asarray(tpr, float))
    if x.shape != y.shape or np.any(np.diff(x) < 0):
        raise ValueError("invalid curve")
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2))
