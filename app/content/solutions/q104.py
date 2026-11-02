import numpy as np


def q104(x, gradient, hessian, damping=1e-06):
    """Apply one damped Newton update using supplied gradient and Hessian callbacks. Solve
    the linear system instead of forming an inverse.
    """
    x = np.asarray(x, float)
    return x - np.linalg.solve(hessian(x) + damping * np.eye(len(x)), gradient(x))
