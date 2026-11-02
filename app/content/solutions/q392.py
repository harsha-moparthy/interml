import numpy as np


def q392(key_buffer, value_buffer, next_position, new_keys, new_values):
    """Append token-major (T,D) keys and values into fixed-capacity ring buffers, in place.
    next_position is the total previously written token count. Return updated count and
    chronological copies of retained entries.
    """
    capacity = len(key_buffer)
    if capacity < 1 or len(value_buffer) != capacity:
        raise ValueError("invalid buffers")
    for k, v in zip(new_keys, new_values, strict=True):
        slot = next_position % capacity
        key_buffer[slot] = k
        value_buffer[slot] = v
        next_position += 1
    positions = np.arange(max(0, next_position - capacity), next_position) % capacity
    return (
        next_position,
        key_buffer[positions].copy(),
        value_buffer[positions].copy(),
    )
