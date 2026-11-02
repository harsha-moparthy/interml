def q421(message_bytes, workers, bandwidth_bytes_per_second, latency_seconds):
    """Estimate ring all-reduce duration with 2*(P-1) latency steps and 2*(P-1)/P times the
    message size transferred per worker.
    """
    if workers < 1 or bandwidth_bytes_per_second <= 0:
        raise ValueError("invalid hardware")
    return (
        2 * (workers - 1) * latency_seconds
        + 2 * (workers - 1) / workers * message_bytes / bandwidth_bytes_per_second
    )
