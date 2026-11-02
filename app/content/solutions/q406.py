import numpy as np


def q406(latencies):
    """Return p50, p95, and p99 latency using NumPy's linear sample-percentile
    interpolation on a nonempty finite array.
    """
    x = np.asarray(latencies, float)
    if not x.size or not np.isfinite(x).all():
        raise ValueError("invalid latencies")
    return dict(zip(("p50", "p95", "p99"), np.percentile(x, [50, 95, 99])))
