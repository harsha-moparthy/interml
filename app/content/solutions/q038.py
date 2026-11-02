import numpy as np


def q038(indices, bins):
    """Count integer indices in [0, bins), including repeated indices correctly."""
    indices = np.asarray(indices, dtype=int)
    if bins < 0 or np.any((indices < 0) | (indices >= bins)):
        raise ValueError("invalid bin index")
    result = np.zeros(bins, dtype=int)
    # Buffered result[indices] += 1 would lose updates for repeated indices.
    np.add.at(result, indices, 1)
    return result
