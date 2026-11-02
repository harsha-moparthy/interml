def q400(arrivals, output_lengths, max_batch):
    """Simulate continuous batching with integer arrival ticks, positive output lengths,
    and one decode token per active request per tick. Admit waiting requests FIFO at
    tick boundaries and return completion times in input order.
    """
    if max_batch < 1 or any((x < 1 for x in output_lengths)):
        raise ValueError("invalid capacity or lengths")
    waiting = sorted(range(len(arrivals)), key=lambda i: (arrivals[i], i))
    active = []
    remaining = list(output_lengths)
    done = [None] * len(arrivals)
    t = 0
    while waiting or active:
        if not active and waiting:
            t = max(t, arrivals[waiting[0]])
        while waiting and arrivals[waiting[0]] <= t and (len(active) < max_batch):
            active.append(waiting.pop(0))
        for i in active:
            remaining[i] -= 1
        for i in active:
            if remaining[i] == 0:
                done[i] = t + 1
        active = [i for i in active if remaining[i] > 0]
        t += 1
    return done
