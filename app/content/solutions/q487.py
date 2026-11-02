def q487(recommended, relevant, k):
    """Compute precision@k and recall@k for a unique ranked list of at least k items. Empty
    relevance sets have recall zero.
    """
    if not 1 <= k <= len(recommended):
        raise ValueError("invalid k")
    relevant = set(relevant)
    hits = len(set(recommended[:k]) & relevant)
    return (hits / k, hits / len(relevant) if relevant else 0.0)
