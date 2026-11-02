import numpy as np


def q465(visit_counts, players_to_move, winner, temperature=1):
    """Build AlphaZero policy targets from nonnegative visit rows and outcome values for
    players +1/-1. winner=0 means draw. Temperature zero gives one-hot argmax;
    empty-visit rows are rejected.
    """
    visits = np.asarray(visit_counts, float)
    if np.any(visits.sum(-1) <= 0) or temperature < 0:
        raise ValueError("invalid visits or temperature")
    if temperature == 0:
        policy = np.eye(visits.shape[1])[visits.argmax(-1)]
    else:
        log = np.full_like(visits, -np.inf)
        np.log(visits, out=log, where=visits > 0)
        log /= temperature
        weights = np.exp(log - log.max(-1, keepdims=True))
        policy = weights / weights.sum(-1, keepdims=True)
    return (policy, np.asarray(players_to_move) * winner)
