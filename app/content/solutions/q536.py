import numpy as np


def q536(rankings, relevance):
    """Return per-query (reciprocal_rank,average_precision), MRR, and MAP. Deduplicate
    ranked IDs. AP denominator is the TOTAL number of relevant documents, including
    unretrieved ones. Queries with empty relevance score zero and remain in the macro
    mean.
    """
    metrics = []
    for ranked, relevant in zip(rankings, relevance, strict=True):
        truth = set(relevant)
        hits = 0
        ap = 0.0
        rr = 0.0
        for rank, doc in enumerate(dict.fromkeys(ranked), 1):
            if doc in truth:
                hits += 1
                ap += hits / rank
                if rr == 0:
                    rr = 1 / rank
        metrics.append((rr, ap / len(truth) if truth else 0.0))
    if not metrics:
        raise ValueError("no queries")
    return (
        metrics,
        float(np.mean([m[0] for m in metrics])),
        float(np.mean([m[1] for m in metrics])),
    )
