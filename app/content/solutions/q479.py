import numpy as np


def q479(steps, offset=0.008, max_beta=0.999):
    """Construct the cosine cumulative-alpha schedule and derive clipped step betas. Return
    betas and the cumulative product of the resulting step alphas.
    """
    if steps < 1:
        raise ValueError("positive steps required")
    t = np.linspace(0, 1, steps + 1)
    cumulative = np.cos((t + offset) / (1 + offset) * np.pi / 2) ** 2
    cumulative /= cumulative[0]
    beta = np.clip(1 - cumulative[1:] / cumulative[:-1], 0, max_beta)
    return (beta, np.cumprod(1 - beta))
