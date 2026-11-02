import torch


@torch.no_grad()
def clip_grad_norm(parameters, max_norm, eps=1e-6):
    """Clip dense gradients jointly in place; return their original global L2 norm.

    Accumulate squared norms in float64. Parameters must share a device.
    Reject nonfinite gradients before mutating any of them.
    """
    gradients = [p.grad for p in parameters if p.grad is not None]
    if max_norm < 0 or eps <= 0:
        raise ValueError("invalid clipping settings")
    if not gradients:
        return torch.tensor(0.0)
    if any(g.is_sparse or g.device != gradients[0].device for g in gradients):
        raise ValueError("expected dense gradients on one device")
    norm = torch.stack([g.double().square().sum() for g in gradients]).sum().sqrt()
    if not torch.isfinite(norm):
        raise ValueError("gradient norm is not finite")
    factor = (max_norm / (norm + eps)).clamp(max=1)
    for gradient in gradients:
        gradient.mul_(factor.to(gradient.dtype))
    return norm
