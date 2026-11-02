import torch


def q052(labels, sample_count, seed=0):
    """Construct a replacement sampler with inverse-class-frequency weights, giving each
    observed class equal expected sampling probability.
    """
    from torch.utils.data import WeightedRandomSampler

    _, inverse, counts = torch.unique(
        torch.as_tensor(labels), return_inverse=True, return_counts=True
    )
    # Inverse class frequency makes total sampling mass equal across observed classes.
    weights = 1.0 / counts[inverse].double()
    return WeightedRandomSampler(
        weights,
        sample_count,
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )
