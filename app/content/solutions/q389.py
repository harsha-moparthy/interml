import numpy as np
from collections import Counter


def q389(
    logits,
    history,
    repetition=1,
    frequency=0,
    presence=0,
    stop_sequences=(),
):
    """Apply sign-aware repetition penalty followed by count and presence penalties. Return
    adjusted logits and whether any nonempty stop-token sequence matches the history
    suffix.
    """
    if repetition <= 0:
        raise ValueError("positive repetition penalty required")
    x = np.asarray(logits, float).copy()
    for token, count in Counter(history).items():
        x[token] = x[token] / repetition if x[token] > 0 else x[token] * repetition
        x[token] -= frequency * count + presence
    stop = any(
        (
            len(s) > 0
            and len(history) >= len(s)
            and (list(history[-len(s) :]) == list(s))
            for s in stop_sequences
        )
    )
    return (x, stop)
