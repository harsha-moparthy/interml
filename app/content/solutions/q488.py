import numpy as np


def q488(relevance, k):
    """Compute NDCG@k from nonnegative relevance grades in ranked order. Use gains
    2^grade-1 and log2(rank+1) discount. Return zero when ideal DCG is zero.
    """
    values = np.asarray(relevance, float)
    k = min(k, len(values))
    if k <= 0:
        return 0.0
    discounts = np.log2(np.arange(k) + 2)
    dcg = ((2 ** values[:k] - 1) / discounts).sum()
    ideal = ((2 ** np.sort(values)[::-1][:k] - 1) / discounts).sum()
    return float(dcg / ideal) if ideal else 0.0
