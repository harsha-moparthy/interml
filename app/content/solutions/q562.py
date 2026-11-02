def q562(loss_vectors, masks):
    """Backpropagate a global VALID-TOKEN mean using precomputed micro-batch per-token loss
    tensors and Boolean masks. Return detached loss. Sum each selected loss then divide
    by total valid tokens; ignore all-masked micro-batches. Caller zeroes gradients and
    steps optimizer. This single-process core does not apply a hidden DDP scale factor.
    """
    if len(loss_vectors) != len(masks):
        raise ValueError("microbatch mismatch")
    count = sum((int(m.sum()) for m in masks))
    if count == 0:
        raise ValueError("no valid tokens")
    value = 0.0
    # Normalize by total valid tokens so unequal microbatches match one combined batch.
    for loss, mask in zip(loss_vectors, masks):
        if mask.any():
            term = loss[mask].sum() / count
            term.backward()
            value += float(term.detach())
    return value
