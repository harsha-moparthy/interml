"""Update a parameter and its momentum velocity using the supplied gradient."""

import numpy as np


def momentum_optimizer(parameter, grad, velocity, learning_rate=0.01, momentum=0.9):
    assert learning_rate > 0, "Learning rate must be positive"
    assert 0 <= momentum < 1, "Momentum must be between 0 and 1"
    velocity = momentum * velocity + learning_rate * grad
    parameter = parameter - velocity
    return (np.round(parameter, 5), np.round(velocity, 5))
