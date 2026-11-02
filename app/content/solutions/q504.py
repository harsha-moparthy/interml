import torch


def q504(center, worker_sums, worker_counts, momentum=0.9):
    """Update a DINO center from worker raw-logit sums [W,K] and sample/view counts [W].
    Return a detached [1,K] center using count-weighted global averaging; zero-count
    workers contribute zero sums. This is a pure collective-reduction reference, not
    distributed communication code.
    """
    counts = torch.as_tensor(worker_counts, device=worker_sums.device)
    if not 0 <= momentum <= 1 or torch.any(counts < 0) or counts.sum() <= 0:
        raise ValueError("invalid momentum/counts")
    with torch.no_grad():
        mean = worker_sums.sum(0, keepdim=True) / counts.sum()
        return momentum * center + (1 - momentum) * mean
