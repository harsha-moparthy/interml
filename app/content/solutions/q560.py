import math
import numpy as np


def q560(task_sizes, draws, exponent=0.5, seed=0):
    """Sample task indices with probabilities proportional to n^exponent, excluding empty
    tasks. Return probabilities, sampled IDs, and realized counts. exponent=0 is uniform
    over nonempty tasks; exponent=1 is example-proportional. Use a seeded NumPy
    generator.
    """
    sizes = np.asarray(task_sizes, float)
    if (
        np.any(sizes < 0)
        or not np.any(sizes > 0)
        or draws < 0
        or (not math.isfinite(exponent))
    ):
        raise ValueError("invalid sizes/draws")
    logits = np.full(len(sizes), -np.inf)
    valid = sizes > 0
    logits[valid] = exponent * np.log(sizes[valid])
    weights = np.exp(logits - np.max(logits))
    weights /= weights.sum()
    ids = np.random.default_rng(seed).choice(len(sizes), size=draws, p=weights)
    return (weights, ids, np.bincount(ids, minlength=len(sizes)))
