import torch


def inverted_dropout(
    x, probability=0.5, *, training=True, drop_path=False, generator=None
):
    """Drop elements, or whole residual paths independently per batch example.

    Apply drop_path=True to the residual branch BEFORE adding the skip connection.
    Evaluation is identity; p=1 returns zeros without dividing by zero.
    """
    if not 0 <= probability <= 1 or x.ndim == 0:
        raise ValueError("expected a batch tensor and probability in [0,1]")
    if not training or probability == 0:
        return x
    if probability == 1:
        return torch.zeros_like(x)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) if drop_path else x.shape
    keep = torch.rand(shape, device=x.device, generator=generator) >= probability
    return x * keep / (1 - probability)
