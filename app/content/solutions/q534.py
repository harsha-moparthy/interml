import torch
from torch.nn import functional as F


def q534(queries, documents, widths=(32, 64, 128), temperature=0.1):
    """Average diagonal-positive in-batch retrieval cross-entropy across independently
    normalized embedding prefixes. Inputs have equal [B,D] shape and widths are positive
    <=D. Return scalar Matryoshka-style training loss; cross-example duplicates need a
    different positive mask.
    """
    if (
        not widths
        or queries.shape != documents.shape
        or min(widths) < 1
        or (max(widths) > queries.shape[1])
        or (temperature <= 0)
    ):
        raise ValueError("invalid widths/shapes")
    labels = torch.arange(len(queries), device=queries.device)
    losses = [
        F.cross_entropy(
            F.normalize(queries[:, :w], dim=-1)
            @ F.normalize(documents[:, :w], dim=-1).T
            / temperature,
            labels,
        )
        for w in widths
    ]
    return torch.stack(losses).mean()
