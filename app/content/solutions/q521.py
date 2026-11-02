def q521(losses, masks, weights):
    """Combine per-example task loss vectors using Boolean availability masks. Return
    weighted sum of each observed task mean and observed counts. An all-missing task
    contributes differentiable zero. Missing losses may be NaN: select before reducing,
    not NaN*zero. Task-loader scheduling is outside this objective core.
    """
    if not losses or not len(losses) == len(masks) == len(weights):
        raise ValueError("task mismatch")
    total = losses[0][:0].sum()
    counts = []
    for values, mask, weight in zip(losses, masks, weights):
        selected = values[mask]
        counts.append(selected.numel())
        total = total + weight * (
            selected.mean() if selected.numel() else values[:0].sum()
        )
    return (total, counts)
