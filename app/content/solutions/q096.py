import numpy as np
from scipy import stats


def q096(success_a, n_a, success_b, n_b):
    """Compute a pooled two-proportion z-test for an A/B experiment. Return B-minus-A lift,
    z-score, and two-sided p-value; define z=0 when both groups have identical
    degenerate outcomes.
    """
    pa, pb = (success_a / n_a, success_b / n_b)
    pooled = (success_a + success_b) / (n_a + n_b)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (pb - pa) / se if se else 0.0
    return (pb - pa, float(z), float(2 * stats.norm.sf(abs(z))))
