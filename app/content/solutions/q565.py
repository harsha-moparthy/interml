import torch


def q565(gradients, clip):
    """Jointly clip each example's gradient across all parameter tensors to positive bound
    C. Inputs are a nonempty list of tensors [B,*parameter_shape]. Return clipped
    tensors and original per-example L2 norms; zero gradients remain zero.
    """
    if not gradients or clip <= 0:
        raise ValueError("invalid gradients/clip")
    batch_size = gradients[0].shape[0]
    if any((gradient.shape[0] != batch_size for gradient in gradients)):
        raise ValueError("batch mismatch")
    norm = torch.sqrt(
        sum(
            (gradient.reshape(batch_size, -1).square().sum(1) for gradient in gradients)
        )
    )
    factor = (clip / norm.clamp_min(1e-12)).clamp(max=1)
    return (
        [
            gradient * factor.reshape((batch_size,) + (1,) * (gradient.ndim - 1))
            for gradient in gradients
        ],
        norm,
    )
