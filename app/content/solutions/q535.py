import torch
from torch.nn import functional as F


def q529(documents):
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


def q527(features, weight, bias, k=None, threshold=0.0):
    if threshold < 0 or (k is not None and k < 0):
        raise ValueError("negative sparsity setting")
    z = F.relu(F.linear(features, weight, bias))
    z = torch.where(z > threshold, z, 0)
    if k is not None:
        indices = torch.argsort(z, dim=-1, descending=True, stable=True)[
            :, : min(k, z.shape[1])
        ]
        mask = torch.zeros_like(z, dtype=torch.bool).scatter_(1, indices, True)
        z = z * mask
    return (z, (z > 0).sum(-1))


def q530(query, index, k=10):
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


def q535(gallery_features, query_features, weight, bias, k_nonzero=4, k_results=5):
    """Encode and index a tiny corpus using a supplied trained nonnegative projection, then
    search query codes. Return sparse codes, postings, rankings, and total nonzeros.
    Gallery IDs are row indices. This is the checkpoint-to-index/inference core;
    optimization and relevance evaluation use earlier exercises.
    """
    with torch.no_grad():
        gallery, _ = q527(gallery_features, weight, bias, k_nonzero)
        queries, _ = q527(query_features, weight, bias, k_nonzero)
    docs = {
        i: [(j, float(v)) for j, v in enumerate(row) if v > 0]
        for i, row in enumerate(gallery)
    }
    index = q529(docs)
    ranks = [
        q530(
            {j: float(v) for j, v in enumerate(row) if v > 0},
            index,
            k_results,
        )
        for row in queries
    ]
    return (gallery, index, ranks, int((gallery > 0).sum()))
