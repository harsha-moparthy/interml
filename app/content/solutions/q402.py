import numpy as np


def q402(prefix_hits, queue_tokens, prompt_length, cache_weight=1):
    """Route to the replica minimizing queue_tokens + prompt_length -
    cache_weight*cached_prefix_tokens. Return the earliest replica index on ties and all
    estimated costs.
    """
    hits = np.asarray(prefix_hits)
    queues = np.asarray(queue_tokens)
    if np.any(hits < 0) or np.any(hits > prompt_length):
        raise ValueError("invalid prefix lengths")
    cost = queues + prompt_length - cache_weight * hits
    return (int(cost.argmin()), cost)
