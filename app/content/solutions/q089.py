import numpy as np


def q089(stream, k, seed=0):
    """Uniformly sample up to k items from a one-pass iterable with reservoir sampling and
    O(k) memory.
    """
    if k < 0:
        raise ValueError("negative sample size")
    rng = np.random.default_rng(seed)
    result = []
    # Each later stream element must have the same final inclusion probability.
    for i, value in enumerate(stream):
        if i < k:
            result.append(value)
        else:
            j = int(rng.integers(i + 1))
            if j < k:
                result[j] = value
    return result
