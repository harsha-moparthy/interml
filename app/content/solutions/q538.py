def q538(rankings, c=60):
    """Reciprocal-rank fusion using one-based ranks and c>0. Within each input list retain
    only the first occurrence of an ID. Return all (ID,score) pairs, breaking ties by
    comparable ID.
    """
    if c <= 0:
        raise ValueError("c must be positive")
    scores = {}
    # Reciprocal-rank fusion uses ranks, allowing incomparable score scales to be combined.
    for ranked in rankings:
        for rank, doc in enumerate(dict.fromkeys(ranked), 1):
            scores[doc] = scores.get(doc, 0) + 1 / (c + rank)
    return sorted(scores.items(), key=lambda p: (-p[1], p[0]))
