import math

import torch


def alibi_bias(num_heads, query_length, key_length, *, query_offset=0, device=None):
    """Return causal ALiBi bias (1,H,L,S), including non-power-of-two head counts.

    Add to scaled QK scores before softmax. -inf excludes future keys.
    Cached queries start at query_offset. Time/storage: O(H*L*S).
    """
    if num_heads <= 0 or min(query_length, key_length, query_offset) < 0:
        raise ValueError("heads must be positive; lengths and offset nonnegative")

    def power_of_two_slopes(count):
        first = 2 ** (-(2 ** -(math.log2(count) - 3)))
        return [first ** (index + 1) for index in range(count)]

    nearest = 2 ** int(math.log2(num_heads))
    slopes = power_of_two_slopes(nearest)
    if nearest != num_heads:
        slopes += power_of_two_slopes(2 * nearest)[::2][: num_heads - nearest]
    slopes = torch.tensor(slopes, device=device)[:, None, None]
    queries = query_offset + torch.arange(query_length, device=device)
    keys = torch.arange(key_length, device=device)
    distance = queries[:, None] - keys[None, :]
    bias = -slopes * distance
    return bias.masked_fill(distance < 0, -torch.inf).unsqueeze(0)
