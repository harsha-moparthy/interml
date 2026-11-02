from torch.nn import functional as F


def q539(scores, labels, query_ids):
    """Compute mean pairwise RankNet logistic loss over same-query pairs with strictly
    ordered labels. Return loss and pair count. All-tied/no-pair inputs return
    differentiable zero; no cross-query pairs are trained.
    """
    mask = (labels[:, None] > labels[None, :]) & (
        query_ids[:, None] == query_ids[None, :]
    )
    margins = (scores[:, None] - scores[None, :])[mask]
    return (
        (F.softplus(-margins).mean() if margins.numel() else scores.sum() * 0),
        margins.numel(),
    )
