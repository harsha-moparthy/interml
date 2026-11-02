import numpy as np


def q215(truth, prediction, groups):
    """Compute demographic-parity and equalized-odds gaps across groups for binary
    outcomes. Require each group to contain positive and negative true labels. Return
    max-minus-min positive-rate, TPR, and FPR gaps.
    """
    y, p, g = (
        np.asarray(truth, bool),
        np.asarray(prediction, bool),
        np.asarray(groups),
    )
    rows = []
    for group in np.unique(g):
        mask = g == group
        positive = mask & y
        negative = mask & ~y
        if not positive.any() or not negative.any():
            raise ValueError("undefined group rates")
        rows.append([p[mask].mean(), p[positive].mean(), p[negative].mean()])
    gaps = np.ptp(rows, axis=0)
    return {
        "demographic_parity": float(gaps[0]),
        "equalized_odds": float(max(gaps[1:])),
        "tpr_gap": float(gaps[1]),
        "fpr_gap": float(gaps[2]),
    }
