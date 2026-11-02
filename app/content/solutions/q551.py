import math


def q551(demand, supply, count, minimum_count=5, threshold=0.1):
    """Compute a toy normalized gap max(demand-supply,0)/max(demand,1), gated by minimum
    support and a supplied threshold. Return score and reason
    ('low_support','below_threshold','gap'). Inputs are finite nonnegative scalars; this
    is an explicit practice rule, not an undocumented business formula.
    """
    if not all(
        (
            math.isfinite(v) and v >= 0
            for v in (demand, supply, count, minimum_count, threshold)
        )
    ):
        raise ValueError("finite nonnegative inputs required")
    gap = max(demand - supply, 0) / max(demand, 1)
    if count < minimum_count:
        return (0.0, "low_support")
    if gap < threshold:
        return (0.0, "below_threshold")
    return (gap, "gap")
