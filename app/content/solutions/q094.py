import numpy as np
from scipy import stats


def q094(values, confidence=0.95):
    """Return a Student-t confidence interval for the population mean from at least two
    independent observations with unknown variance.
    """
    x = np.asarray(values, float)
    half = (
        stats.t.ppf((1 + confidence) / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    )
    return (float(x.mean() - half), float(x.mean() + half))
