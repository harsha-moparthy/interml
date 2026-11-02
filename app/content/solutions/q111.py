import numpy as np


def q111(operation, values, upstream):
    """Return local vector-Jacobian products for scalar add, multiply, and tanh operations.
    Values holds the operation's scalar inputs.
    """
    if operation == "add":
        return (upstream, upstream)
    if operation == "mul":
        return (upstream * values[1], upstream * values[0])
    if operation == "tanh":
        return (upstream * (1 - np.tanh(values[0]) ** 2),)
    raise ValueError("unknown operation")
