import numpy as np


def q478(x_t, predicted_noise, alpha_bar_t, alpha_bar_previous):
    """Compute a deterministic eta=0 DDIM step without clipping predicted clean samples."""
    noise = np.asarray(predicted_noise)
    clean = (np.asarray(x_t) - np.sqrt(1 - alpha_bar_t) * noise) / np.sqrt(alpha_bar_t)
    return np.sqrt(alpha_bar_previous) * clean + np.sqrt(1 - alpha_bar_previous) * noise
