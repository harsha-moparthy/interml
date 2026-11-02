import torch
from torch.nn import functional as F


def q567(clipped, clip, sigma, denominator, generator=None):
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
        output.append((summed + sigma * clip * noise) / denominator)
    return output


def q564(model, x, y, loss_fn):
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not len(x) or not parameters:
        raise ValueError("need examples and parameters")
    rows = []
    for i in range(len(x)):
        loss = loss_fn(model(x[i : i + 1]), y[i : i + 1])
        grads = torch.autograd.grad(loss, parameters, allow_unused=True)
        rows.append(
            [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, grads)]
        )
    return [torch.stack([row[j] for row in rows]) for j in range(len(parameters))]


def q565(gradients, clip):
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


def q573(
    model,
    x,
    y,
    indices,
    optimizer,
    clip,
    sigma,
    denominator,
    generator=None,
):
    """Run ONE selected-subset DP-SGD numerical training step for a classifier, using
    cross-entropy, per-example joint clipping, Gaussian aggregation, and a supplied
    denominator. Return unclipped norms. indices are already selected without
    duplicates. The caller owns private selection, sampling and end-to-end accounting;
    this core makes no epsilon claim.
    """
    if len(set(indices)) != len(indices):
        raise ValueError("duplicate subset IDs")
    parameters = [p for p in model.parameters() if p.requires_grad]
    if indices:
        grads = q564(model, x[indices], y[indices], F.cross_entropy)
        clipped, norms = q565(grads, clip)
    else:
        clipped = [p.new_empty((0, *p.shape)) for p in parameters]
        norms = x.new_empty(0)
    noisy = q567(clipped, clip, sigma, denominator, generator)
    optimizer.zero_grad(set_to_none=True)
    for p, g in zip(parameters, noisy):
        p.grad = g.detach()
    optimizer.step()
    return norms
