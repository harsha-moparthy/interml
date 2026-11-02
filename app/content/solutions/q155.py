import numpy as np


def q155(parameters, evaluate):
    """Enumerate the Cartesian product of a parameter grid, call evaluate(config) returning
    a score to maximize, and return the best configuration and score. Ties keep the
    first configuration.
    """
    import itertools

    names = list(parameters)
    best = None
    score = -np.inf
    for values in itertools.product(*(parameters[n] for n in names)):
        config = dict(zip(names, values))
        current = float(evaluate(config))
        if current > score:
            best, score = (config, current)
    return (best, score)
