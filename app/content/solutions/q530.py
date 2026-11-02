def q530(query, index, k=10):
    """Score a nonnegative sparse query dictionary against inverted postings. Return up to
    k (document_id,score) pairs sorted by decreasing score then ID; return only strictly
    positive matches, with no zero-score fill.
    """
    if k < 0 or any((w < 0 for w in query.values())):
        raise ValueError("invalid query/k")
    scores = {}
    for dim, qweight in query.items():
        for doc, dweight in index.get(dim, []):
            scores[doc] = scores.get(doc, 0) + qweight * dweight
    return sorted(
        ((doc, s) for doc, s in scores.items() if s > 0),
        key=lambda pair: (-pair[1], pair[0]),
    )[:k]
