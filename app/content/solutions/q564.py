import torch


def q564(model, x, y, loss_fn):
    """Return a list of per-example gradients, one [B,*parameter.shape] tensor per
    trainable parameter. Evaluate one example at a time using a scalar loss_fn; unused
    parameters get zeros. The model must have no cross-example operations (e.g. training
    BatchNorm). This intentionally slow autograd reference does not populate parameter
    .grad.
    """
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
