def q529(documents):
    """Build inverted postings from {document_id: [(dimension,nonnegative_weight),...]}.
    Sum duplicate coordinates, remove zeros, and sort each posting by comparable
    document IDs. Return {dimension:[(id,weight),...]}. IDs and dimensions must be
    hashable.
    """
    index = {}
    for doc, pairs in documents.items():
        summed = {}
        for dim, weight in pairs:
            if weight < 0:
                raise ValueError("nonnegative weights required")
            summed[dim] = summed.get(dim, 0) + weight
        for dim, weight in summed.items():
            if weight:
                index.setdefault(dim, []).append((doc, weight))
    return {dim: sorted(postings) for dim, postings in index.items()}
