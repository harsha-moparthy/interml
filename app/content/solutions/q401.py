def q401(prefill_remaining, decode_requests, token_budget, chunk_size):
    """Schedule one iteration: reserve one token per decode request first, then allocate
    FIFO prefill chunks. Return decode count, per-request prefill allocations, and
    updated prefill remainders.
    """
    if min(token_budget, decode_requests) < 0 or chunk_size < 1:
        raise ValueError("invalid budget")
    decoded = min(decode_requests, token_budget)
    left = token_budget - decoded
    allocation = []
    for n in prefill_remaining:
        take = min(n, chunk_size, left)
        allocation.append(take)
        left -= take
    return (
        decoded,
        allocation,
        [n - a for n, a in zip(prefill_remaining, allocation)],
    )
