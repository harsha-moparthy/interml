import numpy as np


def q481(velocity, initial_noise, steps):
    """Integrate dx/dt=velocity(x,t) from t=0 noise to t=1 data with uniform explicit Euler
    steps. Return the final NumPy state.
    """
    if steps < 1:
        raise ValueError("positive steps required")
    x = np.array(initial_noise, float, copy=True)
    dt = 1 / steps
    for i in range(steps):
        x = x + dt * velocity(x, i * dt)
    return x
