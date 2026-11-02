import torch


def q042(values):
    """Use autograd to return the gradient of sum(x^3) at supplied floating-point values,
    without accumulating into an existing graph.
    """
    # Detach and clone so differentiation cannot modify the caller’s graph or storage.
    x = torch.as_tensor(values, dtype=torch.float64).detach().clone().requires_grad_()
    return torch.autograd.grad((x**3).sum(), x)[0]
