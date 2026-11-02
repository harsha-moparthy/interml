import math
import numpy as np


def q544(candidates, relevance, k=10):
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


def q545(queries, retrieve, rerank, relevance, candidate_count=20, k=10):
    """Run a small two-stage evaluation with supplied retrieve(query,N)->IDs and
    rerank(query,IDs)->scores callbacks. Deduplicate candidates, validate score
    alignment/finite values, stable-sort by reranker score, and return query results
    plus mean actual NDCG. No fitting is performed; callbacks must already be trained
    without test-label access.
    """
    results = []
    for query, truth in zip(queries, relevance, strict=True):
        ids = list(dict.fromkeys(retrieve(query, candidate_count)))[:candidate_count]
        scores = list(rerank(query, ids))
        if len(scores) != len(ids) or not all((math.isfinite(s) for s in scores)):
            raise ValueError("invalid reranker scores")
        order = sorted(range(len(ids)), key=lambda i: (-scores[i], i))
        ranked = [ids[i] for i in order]
        results.append({"ids": ranked, "metrics": q544(ranked, truth, k)})
    return (
        results,
        (float(np.mean([r["metrics"][2] for r in results])) if results else 0.0),
    )
