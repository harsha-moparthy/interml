"""Apply an Adagrad update and return the updated squared-gradient accumulator."""

import numpy as np


def adagrad_optimizer(parameter, grad, G, learning_rate=0.01, epsilon=1e-08):
    assert learning_rate > 0, "Learning rate must be positive"
    assert epsilon > 0, "Epsilon must be positive"
    assert (
        all(G >= 0) if isinstance(G, np.ndarray) else G >= 0
    ), "G must be non-negative"
    G = G + grad**2
    update = learning_rate * grad / (np.sqrt(G) + epsilon)
    parameter = parameter - update
    return (np.round(parameter, 5), np.round(G, 5))
