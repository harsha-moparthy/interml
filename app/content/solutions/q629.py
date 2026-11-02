import math


def learning_rate(step, *, total_steps, warmup_steps, peak_lr, minimum_lr=0.0):
    """Zero-based update schedule; warmup reaches peak, final update reaches minimum.

    Apply the returned rate BEFORE optimizer.step(). Past the budget, hold minimum.
    At least two decay updates are required to define both cosine endpoints.
    """
    if (
        step < 0
        or warmup_steps < 0
        or total_steps - warmup_steps < 2
        or not 0 <= minimum_lr <= peak_lr
    ):
        raise ValueError("invalid schedule")
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps - 1))
    return minimum_lr + (peak_lr - minimum_lr) * (1 + math.cos(math.pi * progress)) / 2
