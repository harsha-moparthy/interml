import torch
from torch.nn import functional as F


def q540(token_features, term_ids, weight, bias, vocabulary):
    """Predict nonnegative token impacts by a supplied linear head, then SUM impacts for
    repeated term IDs into [B,V]. Negative term IDs denote padding/special tokens. This
    explicit sum-pooling variant is a small learned-impact core, not a full DeepImpact
    model.
    """
    impact = F.relu(F.linear(token_features, weight, bias)).squeeze(-1)
    valid = term_ids >= 0
    if torch.any(term_ids[valid] >= vocabulary):
        raise ValueError("term outside vocabulary")
    output = impact.new_zeros((len(impact), vocabulary))
    return output.scatter_add(1, term_ids.clamp_min(0), torch.where(valid, impact, 0))
