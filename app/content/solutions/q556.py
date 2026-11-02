import torch
from torch.nn import functional as F


def q556(x, base_weight, adapters, adapter_ids):
    """Apply one selected LoRA update per row: base linear plus scale*(x@A.T)@B.T. adapters
    maps ID to (A[r,in],B[out,r],scale); base_weight is [out,in]. Return [batch,out],
    preserve row order, and detach the base weight. Unknown IDs raise. Only selected
    adapters receive gradients.
    """
    if len(x) != len(adapter_ids):
        raise ValueError("adapter IDs length")
    return torch.stack(
        [
            F.linear(row, base_weight.detach()) + scale * F.linear(F.linear(row, a), b)
            for row, key in zip(x, adapter_ids)
            for a, b, scale in [adapters[key]]
        ]
    )
