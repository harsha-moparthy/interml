import numpy as np


def q304(tokens):
    """Construct next-token prediction inputs and targets by shifting a one-dimensional
    token sequence by one position.
    """
    tokens = np.asarray(tokens, dtype=int)
    return (tokens[:-1], tokens[1:])
