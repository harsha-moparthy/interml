import torch
from torch.nn import functional as F


def q525(query, documents, query_mask, document_mask):
    """ColBERT MaxSim: query [B,Q,D], documents [B,N,L,D], masks [B,Q]/[B,N,L]. Normalize
    tokens, sum each valid query token's maximum document-token similarity. Empty
    documents score -inf; an empty query scores zero against nonempty documents.
    """
    q = F.normalize(query, dim=-1)
    d = F.normalize(documents, dim=-1)
    sim = torch.einsum("bqd,bnld->bnql", q, d)
    sim = sim.masked_fill(~document_mask[:, :, None, :], -torch.inf)
    best = sim.max(-1).values
    scores = torch.where(query_mask[:, None, :], best, 0).sum(-1)
    return scores.masked_fill(~document_mask.any(-1), -torch.inf)
