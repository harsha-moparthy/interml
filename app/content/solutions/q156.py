import numpy as np


def q156(parameters, evaluate, iterations, seed=0):
    """Sample configurations independently from finite lists with replacement, evaluate
    each, and return the highest-scoring configuration.
    """
    rng = np.random.default_rng(seed)
    best = None
    score = -np.inf
    for _ in range(iterations):
        config = {
            k: values[int(rng.integers(len(values)))]
            for k, values in parameters.items()
        }
        current = float(evaluate(config))
        if current > score:
            best, score = (config, current)
    return (best, score)
