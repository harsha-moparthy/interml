import numpy as np


def q212(latencies, success, duration):
    """Return request count, success rate, requests per second, and p50/p95/p99 latency
    from a nonempty set of observed requests over a positive duration.
    """
    x = np.asarray(latencies, float)
    return {
        "requests": len(x),
        "success_rate": float(np.mean(success)),
        "throughput": len(x) / duration,
        "latency_percentiles": np.percentile(x, [50, 95, 99]),
    }
