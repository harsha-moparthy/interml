import torch


def q574(hidden, mask, mode="mean"):
    """Extract detached frozen-feature vectors from supplied [B,T,D] BERT hidden states and
    Boolean [B,T] masks. Support CLS or masked mean pooling; reject all-padding examples
    and an invalid CLS. Caller runs the encoder in eval mode; this is
    pooling/cache-payload preparation, not a persistent cache.
    """
    if hidden.shape[:2] != mask.shape or not mask.any(-1).all():
        raise ValueError("invalid mask")
    if mode == "cls":
        if not mask[:, 0].all():
            raise ValueError("CLS is masked")
        return hidden[:, 0].detach().clone()
    if mode != "mean":
        raise ValueError("unknown pooling")
    selected = torch.where(mask[:, :, None], hidden, 0)
    return (selected.sum(1) / mask.sum(1, keepdim=True)).detach()
