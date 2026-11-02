import numpy as np


def q475(steps, beta_start=0.0001, beta_end=0.02):
    """Create a linear DDPM beta schedule and return beta, alpha=1-beta, and cumulative
    alpha products.
    """
    if steps < 1 or not 0 < beta_start <= beta_end < 1:
        raise ValueError("invalid schedule")
    beta = np.linspace(beta_start, beta_end, steps)
    alpha = 1 - beta
    return (beta, alpha, np.cumprod(alpha))
