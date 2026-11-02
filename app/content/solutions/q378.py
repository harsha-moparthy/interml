import numpy as np


def q378(matches, model_count):
    """Aggregate pairwise judgments (model_a,model_b,score_a) into win rates, counting ties
    as 0.5. Return descending stable ranking and rates; unrated models have NaN rates.
    """
    points = np.zeros(model_count)
    games = np.zeros(model_count)
    for a, b, score in matches:
        points[a] += score
        points[b] += 1 - score
        games[a] += 1
        games[b] += 1
    rates = np.divide(points, games, out=np.full(model_count, np.nan), where=games > 0)
    return (np.argsort(-np.nan_to_num(rates, nan=-1), kind="stable"), rates)
