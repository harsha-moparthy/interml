import numpy as np


def q333(inverse_frequencies, factor, low_index, high_index):
    """Implement YaRN-style frequency blending with supplied correction-band indices:
    retain high-frequency pairs, interpolate to frequencies divided by factor, and
    return the attention magnitude scale 1+0.1*log(factor). Band endpoints are
    caller-provided.
    """
    frequency = np.asarray(inverse_frequencies, float)
    if factor < 1 or high_index <= low_index:
        raise ValueError("invalid scaling band")
    ramp = np.clip(
        (np.arange(len(frequency)) - low_index) / (high_index - low_index),
        0,
        1,
    )
    return (
        frequency * (1 - ramp) + frequency / factor * ramp,
        float(1 + 0.1 * np.log(factor)),
    )
