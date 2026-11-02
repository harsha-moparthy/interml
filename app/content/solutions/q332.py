import numpy as np


def q332(
    inverse_frequencies,
    original_context=8192,
    factor=8,
    low_factor=1,
    high_factor=4,
):
    """Rescale RoPE inverse frequencies with the Llama 3.1 wavelength rule: retain short
    wavelengths, divide long wavelengths by factor, and smoothly interpolate the middle
    band.
    """
    frequency = np.asarray(inverse_frequencies, float)
    wavelength = 2 * np.pi / frequency
    low = original_context / low_factor
    high = original_context / high_factor
    smooth = (original_context / wavelength - low_factor) / (high_factor - low_factor)
    interpolated = (1 - smooth) * frequency / factor + smooth * frequency
    return np.where(
        wavelength > low,
        frequency / factor,
        np.where(wavelength < high, frequency, interpolated),
    )
