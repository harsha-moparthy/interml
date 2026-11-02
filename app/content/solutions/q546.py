import numpy as np


def q546(actual, predicted, groups=None):
    """Return pooled WAPE, per-group WAPE, and unweighted mean of defined group WAPEs. Mask
    rows unless both values are finite; numerator and denominator use the SAME mask. A
    zero absolute-target denominator is NaN. Values are ratios, not percentages.
    """
    y = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("expected matching vectors")
    g = np.zeros(len(y), int) if groups is None else np.asarray(groups)
    if len(g) != len(y):
        raise ValueError("group length")
    valid = np.isfinite(y) & np.isfinite(p)

    def score(mask):
        denominator = np.abs(y[mask]).sum()
        return (
            np.abs(y[mask] - p[mask]).sum() / denominator
            if denominator
            else float("nan")
        )

    values = {key: score(valid & (g == key)) for key in dict.fromkeys(g.tolist())}
    defined = [v for v in values.values() if np.isfinite(v)]
    return (
        score(valid),
        values,
        float(np.mean(defined)) if defined else float("nan"),
    )
