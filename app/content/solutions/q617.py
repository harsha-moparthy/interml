import torch


def sliding_window_mask(query_length, key_length, window, *, offset=0, device=None):
    """Return (L,S) boolean permissions for a causal window INCLUDING the current token.

    Query positions start at offset; keys start at zero. True means allowed.
    A dense mask costs O(L*S) space; a sparse kernel is needed for O(L*window) work.
    """
    if min(query_length, key_length, offset) < 0 or window < 1:
        raise ValueError("nonnegative lengths/offset and positive window required")
    distance = (
        offset
        + torch.arange(query_length, device=device)[:, None]
        - torch.arange(key_length, device=device)[None, :]
    )
    return (distance >= 0) & (distance < window)
