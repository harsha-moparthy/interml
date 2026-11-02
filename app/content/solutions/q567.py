import torch


def q567(clipped, clip, sigma, denominator, generator=None):
    """Sum supplied already-clipped per-example gradient tensors, add independent Gaussian
    noise with coordinate standard deviation sigma*C, and divide by positive supplied
    denominator. Empty batches still add noise. For Poisson sampling the caller supplies
    expected-batch normalization; this numerical core does not establish a privacy
    guarantee or secure randomness.
    """
    if clip <= 0 or sigma < 0 or denominator <= 0:
        raise ValueError("invalid noise/normalization")
    output = []
    for g in clipped:
        summed = g.sum(0)
        noise = torch.randn(
            summed.shape,
            dtype=summed.dtype,
            device=summed.device,
            generator=generator,
        )
        # Add noise to the sum before dividing by the public normalizer.
        output.append((summed + sigma * clip * noise) / denominator)
    return output
