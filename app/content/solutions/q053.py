from torch.nn import functional as F


def q053(model, x, target, optimizer):
    """Run one supervised classification step: clear gradients, compute mean cross-entropy,
    backpropagate, update parameters, and return detached loss.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(x), target)
    loss.backward()
    optimizer.step()
    return float(loss.detach())
