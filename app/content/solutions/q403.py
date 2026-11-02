def q403(arrivals, prefill_times, transfer_times, decode_times):
    """Simulate one FCFS prefill worker, independent per-request transfers, and one decode
    worker serving requests in original arrival order. Return prefill end,
    transfer-ready, and decode-end timestamps; arrivals must be sorted.
    """
    p_end = 0.0
    d_end = 0.0
    results = []
    for arrival, p, transfer, d in zip(
        arrivals, prefill_times, transfer_times, decode_times, strict=True
    ):
        p_end = max(p_end, arrival) + p
        ready = p_end + transfer
        d_end = max(d_end, ready) + d
        results.append((p_end, ready, d_end))
    return results
