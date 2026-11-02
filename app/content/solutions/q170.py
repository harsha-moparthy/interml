import numpy as np


def q170(bins, gradients, hessians, bin_count, regularization=1):
    """Find the best histogram split of one feature using gradient/Hessian sums. Return a
    cut bin and regularized gain. Both children must contain observations.
    """
    bins = np.asarray(bins, int)
    g = np.bincount(bins, weights=gradients, minlength=bin_count)
    h = np.bincount(bins, weights=hessians, minlength=bin_count)
    count = np.bincount(bins, minlength=bin_count)
    gl = np.cumsum(g)[:-1]
    hl = np.cumsum(h)[:-1]
    gr = g.sum() - gl
    hr = h.sum() - hl
    counts = np.cumsum(count)[:-1]
    gain = 0.5 * (
        gl * gl / (hl + regularization)
        + gr * gr / (hr + regularization)
        - g.sum() ** 2 / (h.sum() + regularization)
    )
    gain[(counts == 0) | (counts == len(bins))] = -np.inf
    index = int(np.argmax(gain))
    return (index, float(gain[index]))
