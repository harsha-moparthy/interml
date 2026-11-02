import numpy as np
from scipy import stats


def q093(a, b):
    """Compute Welch's two-sided t-test for independent samples with at least two
    observations and positive estimated standard error. Return statistic, degrees of
    freedom, and p-value.
    """
    a, b = (np.asarray(a, float), np.asarray(b, float))
    va, vb = (a.var(ddof=1) / len(a), b.var(ddof=1) / len(b))
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va * va / (len(a) - 1) + vb * vb / (len(b) - 1))
    return (float(t), float(df), float(2 * stats.t.sf(abs(t), df)))
