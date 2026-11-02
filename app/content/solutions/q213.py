import numpy as np


def q213(
    baseline_errors,
    baseline_latencies,
    canary_errors,
    canary_latencies,
    max_error_increase=0.01,
    max_latency_ratio=1.1,
):
    """Evaluate a canary rollout with explicit thresholds. Return whether its error-rate
    increase and p95 latency ratio stay within limits, plus both measurements.
    """
    increase = float(np.mean(canary_errors) - np.mean(baseline_errors))
    base = np.percentile(baseline_latencies, 95)
    current = np.percentile(canary_latencies, 95)
    ratio = float(current / base) if base else 1.0 if current == 0 else np.inf
    return (
        increase <= max_error_increase and ratio <= max_latency_ratio,
        increase,
        ratio,
    )
