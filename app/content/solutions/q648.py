import numpy as np


def ranking_metrics(relevance, k):
    """Return mean NDCG@k and MRR@k for rows already in predicted ranking order.

    Supply ALL judged candidates per row so ideal DCG sees the full candidate set.
    Gains use 2**relevance-1; MRR treats positive relevance as relevant. Zero-relevance
    queries contribute zero, rather than being silently dropped from the average.
    """
    relevance = np.asarray(relevance, dtype=float)
    if relevance.ndim != 2 or not len(relevance) or k < 1 or np.any(relevance < 0):
        raise ValueError("nonnegative query-by-candidate relevance required")
    k = min(k, relevance.shape[1])
    discount = 1 / np.log2(np.arange(k) + 2)
    dcg = ((2 ** relevance[:, :k] - 1) * discount).sum(-1)
    ideal = ((2 ** np.sort(relevance, axis=-1)[:, ::-1][:, :k] - 1) * discount).sum(-1)
    ndcg = np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)
    reciprocal = np.where(relevance[:, :k] > 0, 1 / (np.arange(k) + 1), 0).max(
        -1, initial=0
    )
    return float(ndcg.mean()), float(reciprocal.mean())
