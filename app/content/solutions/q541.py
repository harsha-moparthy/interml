import numpy as np


def q541(features, weights, query_ids, k=10):
    """Score supplied unigram/ordered-bigram/unordered-window feature triples by a linear
    weight vector, then rank rows within each query. Return query->row-ID list. Feature
    extraction and validation-only weight fitting are outside this simplified MRF
    scoring core.
    """
    scores = np.asarray(features, float) @ np.asarray(weights, float)
    groups = {}
    for row, query in enumerate(query_ids):
        groups.setdefault(query, []).append(row)
    return {
        q: sorted(rows, key=lambda i: (-scores[i], i))[:k] for q, rows in groups.items()
    }
