import numpy as np
from collections import Counter


def q380(labels_a, labels_b, first_position_wins):
    """Compute Cohen's kappa for paired categorical judgments and the first-position win
    fraction for Boolean decisive judgments. Return NaN when a denominator is undefined.
    """
    a = list(labels_a)
    b = list(labels_b)
    if not a or len(a) != len(b):
        raise ValueError("paired labels required")
    ca, cb = (Counter(a), Counter(b))
    n = len(a)
    observed = sum((x == y for x, y in zip(a, b))) / n
    expected = sum((ca[x] * cb[x] for x in ca.keys() | cb.keys())) / n**2
    kappa = (observed - expected) / (1 - expected) if expected < 1 else float("nan")
    return (
        kappa,
        (
            float(np.mean(first_position_wins))
            if len(first_position_wins)
            else float("nan")
        ),
    )
