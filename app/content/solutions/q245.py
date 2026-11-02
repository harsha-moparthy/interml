"""Apply a Nesterov momentum update using the gradient at the look-ahead position."""

import numpy as np


def nag_optimizer(parameter, grad_fn, velocity, learning_rate=0.01, momentum=0.9):
    assert 0 <= momentum < 1, "Momentum must be between 0 and 1"
    assert learning_rate > 0, "Learning rate must be positive"
    look_ahead = parameter - momentum * velocity
    grad = grad_fn(look_ahead)
    velocity = momentum * velocity + learning_rate * grad
    parameter = parameter - velocity
    return (np.round(parameter, 5), np.round(velocity, 5))
