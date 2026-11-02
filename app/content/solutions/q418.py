import torch


def q418(x, scale, qmin=-127, qmax=127):
    """Implement Torch fake quantization with identity straight-through gradient inside the
    representable range and zero gradient outside it. scale is a fixed positive scalar.
    """
    if scale <= 0:
        raise ValueError("positive scale required")
    clipped = x.clamp(qmin * scale, qmax * scale)
    rounded = torch.round(clipped / scale) * scale
    return clipped + (rounded - clipped).detach()
