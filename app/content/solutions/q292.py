import numpy as np


def q292(sequences, length, pad_id=0):
    """Right-truncate and right-pad token sequences to a fixed nonnegative length. Return
    an integer matrix and clipped original lengths.
    """
    if length < 0:
        raise ValueError("negative length")
    result = np.full((len(sequences), length), pad_id, dtype=int)
    for i, sequence in enumerate(sequences):
        result[i, : min(length, len(sequence))] = sequence[:length]
    return (result, np.array([min(length, len(s)) for s in sequences]))
