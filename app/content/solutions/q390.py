import numpy as np


def q390(logits, state, transitions):
    """Mask logits using a finite-state token grammar represented as
    transitions[state][token]=next_state. Return masked logits; reject states with no
    allowed tokens.
    """
    x = np.full_like(np.asarray(logits, float), -np.inf)
    allowed = transitions.get(state, {})
    if not allowed:
        raise ValueError("no valid next token")
    for token in allowed:
        x[token] = logits[token]
    return x
