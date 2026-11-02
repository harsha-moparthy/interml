import numpy as np


def q569(similarities, k):
    """Greedily maximize facility-location coverage for nonnegative similarities
    [candidates,reference_items]. Select k<=N unique rows with stable tie-breaking;
    maintain each reference item's best similarity. Return selected IDs and objective
    history. This is a separate submodular objective, not a claim about arbitrary
    GLISTER utilities.
    """
    s = np.asarray(similarities, float)
    if s.ndim != 2 or np.any(s < 0) or (not 0 <= k <= len(s)):
        raise ValueError("invalid similarities/k")
    best = np.zeros(s.shape[1])
    selected = []
    history = []
    for _ in range(k):
        gains = np.maximum(s, best).sum(1) - best.sum()
        gains[selected] = -np.inf
        i = int(np.argmax(gains))
        selected.append(i)
        best = np.maximum(best, s[i])
        history.append(float(best.sum()))
    return (selected, history)
