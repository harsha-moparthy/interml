import math
from torch.nn import functional as F


def q502(logits, center, temperature=0.04):
    """Return detached original-DINO teacher targets for logits [2,B,K] and center [1,K].
    Use float32 centered/sharpened softmax; temperature must be positive and finite.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("invalid temperature")
    return F.softmax(
        (logits.detach().float() - center.detach().float()) / temperature,
        dim=-1,
    )
