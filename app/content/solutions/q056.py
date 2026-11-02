from torch.nn import functional as F


def q056(model, batches, optimizer):
    """Accumulate classification gradients over unequal-sized microbatches, matching one
    mean loss over all examples. Update once and return the mean loss. Assume no
    batch-dependent layers.
    """
    batches = list(batches)
    total = sum((len(y) for x, y in batches))
    if total == 0:
        raise ValueError("empty batch")
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    # Weight by example count, not microbatch count, because batch sizes can differ.
    for x, y in batches:
        loss = F.cross_entropy(model(x), y, reduction="sum") / total
        loss.backward()
        loss_sum += float(loss.detach())
    optimizer.step()
    return loss_sum
