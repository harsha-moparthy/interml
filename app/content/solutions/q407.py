import numpy as np


def q407(latencies, duration, sla):
    """For completed requests in an observation window, report request throughput,
    per-request goodput under a latency bound, p99 latency, and whether the aggregate
    p99 SLA holds.
    """
    if duration <= 0 or not len(latencies):
        raise ValueError("positive duration and requests required")
    x = np.asarray(latencies, float)
    p99 = float(np.percentile(x, 99))
    return {
        "throughput": len(x) / duration,
        "goodput": float(np.sum(x <= sla) / duration),
        "p99": p99,
        "p99_sla_met": p99 <= sla,
    }
