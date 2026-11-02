import numpy as np


def q587(baseline, new):
    """Compare repair outcomes in {benchmark_id:bool} mappings on their intersection.
    Return counts, rates, percentage-point change, relative improvement, and unmatched
    IDs. Zero baseline makes relative improvement NaN; no overlap raises. Caller ensures
    matching verifier/token budgets.
    """
    ids = set(baseline) & set(new)
    if not ids:
        raise ValueError("no common benchmark IDs")
    a = sum((bool(baseline[i]) for i in ids))
    b = sum((bool(new[i]) for i in ids))
    n = len(ids)
    return {
        "n": n,
        "baseline_count": a,
        "new_count": b,
        "baseline_rate": a / n,
        "new_rate": b / n,
        "percentage_points": 100 * (b - a) / n,
        "relative_improvement": (b - a) / a if a else np.nan,
        "only_baseline": sorted(set(baseline) - ids),
        "only_new": sorted(set(new) - ids),
    }
