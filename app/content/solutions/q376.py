def q376(rating_a, rating_b, score_a, k=32):
    """Apply a simultaneous Elo update with a 400-point logistic scale. score_a is 1 for a
    win, 0.5 for a draw, or 0 for a loss.
    """
    expected = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    change = k * (score_a - expected)
    return (rating_a + change, rating_b - change)
