import numpy as np


def q474(clean, noise, alpha_bar):
    """Compute forward diffusion x_t=sqrt(alpha_bar)*x_0+sqrt(1-alpha_bar)*noise. alpha_bar
    must be broadcastable to the NumPy input shape.
    """
    return np.sqrt(alpha_bar) * np.asarray(clean) + np.sqrt(
        1 - np.asarray(alpha_bar)
    ) * np.asarray(noise)
