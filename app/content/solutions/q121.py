import numpy as np


def q121(values, ordered_categories):
    """Encode categories using the caller's explicit order; encode unknown categories as -1
    and reject duplicate vocabulary entries.
    """
    mapping = {value: i for i, value in enumerate(ordered_categories)}
    if len(mapping) != len(ordered_categories):
        raise ValueError("duplicate category")
    return np.array([mapping.get(v, -1) for v in values])
