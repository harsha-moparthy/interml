import math


def q572(
    accountant,
    target,
    low=0.0001,
    high=1.0,
    tolerance=1e-05,
    max_sigma=10000.0,
):
    """Find the smallest approximate noise sigma meeting target epsilon via a supplied
    positive monotone-decreasing accountant(sigma). Bracket then bisect; return a
    feasible upper endpoint. The caller fixes sampling rate, steps, delta and adjacency
    in the accountant. Reject nonfinite evaluations/unreachable targets; this is not a
    mock privacy accountant.
    """
    if target <= 0 or not 0 < low < high <= max_sigma or tolerance <= 0:
        raise ValueError("invalid bounds")

    def evaluate(sigma):
        value = float(accountant(sigma))
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid accountant result")
        return value

    if evaluate(low) <= target:
        return low
    while evaluate(high) > target:
        if high >= max_sigma:
            raise ValueError("target not bracketed")
        low = high
        high = min(2 * high, max_sigma)
    for _ in range(200):
        if high - low <= tolerance:
            break
        mid = (low + high) / 2
        if evaluate(mid) <= target:
            high = mid
        else:
            low = mid
    return high
