import numpy as np


def q210(missing, invalid, duplicate_rows):
    """Compute a transparent data-quality summary from Boolean cell masks and a Boolean
    duplicate-row mask: missing rate, invalid rate, duplicate rate, and one minus their
    equally weighted mean.
    """
    rates = np.array(
        [np.mean(missing), np.mean(invalid), np.mean(duplicate_rows)], float
    )
    return {
        "missing_rate": rates[0],
        "invalid_rate": rates[1],
        "duplicate_rate": rates[2],
        "quality": float(1 - rates.mean()),
    }
