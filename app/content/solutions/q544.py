import math


def q544(candidates, relevance, k=10):
    """For one query return candidate recall, oracle candidate-constrained NDCG@k, and
    actual NDCG@k. Relevance maps full-corpus IDs to nonnegative graded labels.
    Deduplicate candidates; denominator uses ideal FULL judged-corpus gains, not only
    retrieved candidates. No-positive queries return zeros.
    """
    if k < 1:
        raise ValueError("k must be positive")
    ids = list(dict.fromkeys(candidates))
    positive = {d for d, y in relevance.items() if y > 0}

    def dcg(grades):
        return sum(((2**y - 1) / math.log2(i + 2) for i, y in enumerate(grades[:k])))

    ideal = dcg(sorted(relevance.values(), reverse=True))
    grades = [relevance.get(d, 0) for d in ids]
    recall = len(set(ids) & positive) / len(positive) if positive else 0.0
    return (
        recall,
        dcg(sorted(grades, reverse=True)) / ideal if ideal else 0.0,
        dcg(grades) / ideal if ideal else 0.0,
    )
