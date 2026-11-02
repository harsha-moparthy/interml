import numpy as np


def q305(tokens, context, stride=1):
    """Create fixed-length next-token training windows. Each input window has context
    tokens and its target window is shifted by one. Keep complete windows only.
    """
    if context <= 0 or stride <= 0:
        raise ValueError("positive sizes required")
    tokens = np.asarray(tokens, int)
    starts = range(0, len(tokens) - context, stride)
    x = [tokens[s : s + context] for s in starts]
    y = [tokens[s + 1 : s + context + 1] for s in starts]
    return (
        np.asarray(x, dtype=int).reshape(-1, context),
        np.asarray(y, dtype=int).reshape(-1, context),
    )
