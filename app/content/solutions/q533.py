import numpy as np


def q533(rankings, relevant, gallery_labels, query_labels, k=5, self_ids=None):
    """Return macro retrieval recall@k and 1-NN class accuracy. Rankings are lists of IDs;
    relevant is one set per query; gallery_labels maps IDs to classes.
    Deduplicate/exclude self before truncation. Empty relevance contributes zero recall
    and empty rankings count as incorrect classification.
    """
    if k < 1 or not rankings:
        raise ValueError("nonempty queries and k>=1 required")
    recall = []
    correct = []
    for i, ranked in enumerate(rankings):
        ids = list(dict.fromkeys(ranked))
        if self_ids is not None:
            ids = [x for x in ids if x != self_ids[i]]
        truth = set(relevant[i])
        top = ids[:k]
        recall.append(len(set(top) & truth) / len(truth) if truth else 0)
        correct.append(bool(ids) and gallery_labels[ids[0]] == query_labels[i])
    return (float(np.mean(recall)), float(np.mean(correct)))
