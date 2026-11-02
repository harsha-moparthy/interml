import numpy as np


def viterbi(log_initial, log_transition, log_emission):
    """Most likely HMM path from log scores: initial(S), transition(S,S), emission(T,S).

    transition[i,j] is from state i to j. -inf permits impossible transitions.
    Return path and its joint log score. Complexity O(T*S²), backpointer space O(T*S).
    For a CRF use unnormalized scores and add any final-state scores before argmax.
    """
    initial = np.asarray(log_initial, dtype=float)
    transition = np.asarray(log_transition, dtype=float)
    emission = np.asarray(log_emission, dtype=float)
    states = len(initial)
    if (
        initial.ndim != 1
        or not states
        or transition.shape != (states, states)
        or emission.ndim != 2
        or emission.shape[1] != states
    ):
        raise ValueError("invalid HMM shapes")
    if any(
        np.isnan(a).any() or np.isposinf(a).any()
        for a in (initial, transition, emission)
    ):
        raise ValueError("scores may be finite or negative infinity")
    if not len(emission):
        return np.array([], dtype=int), 0.0
    scores = initial + emission[0]
    backpointers = np.zeros(emission.shape, dtype=int)
    for step in range(1, len(emission)):
        candidates = scores[:, None] + transition
        backpointers[step] = candidates.argmax(axis=0)
        scores = candidates.max(axis=0) + emission[step]
    if not np.isfinite(scores.max()):
        raise ValueError("no valid state path")
    path = np.empty(len(emission), dtype=int)
    path[-1] = scores.argmax()
    for step in range(len(emission) - 1, 0, -1):
        path[step - 1] = backpointers[step, path[step]]
    return path, float(scores.max())
