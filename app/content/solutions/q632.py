from itertools import islice

import torch
import torch.nn.functional as F


def train_epoch(
    model,
    batches,
    optimizer,
    *,
    accumulation_steps=4,
    device="cpu",
    scaler=None,
    max_norm=1.0,
):
    """Classification training with unequal microbatches and a final partial group.

    Return mean example loss and the scaler, which should be reused across epochs.
    Uses fp16+scaling on CUDA, bfloat16 autocast on CPU. With BatchNorm/dropout,
    accumulation need not match one large forward pass even when gradients are weighted.
    """
    device = torch.device(device)
    if accumulation_steps < 1 or device.type not in ("cpu", "cuda"):
        raise ValueError("positive accumulation steps and CPU/CUDA device required")
    scaler = (
        scaler
        if scaler is not None
        else torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    )
    model.train()
    iterator = iter(batches)
    total_loss, total_examples = 0.0, 0
    while group := list(islice(iterator, accumulation_steps)):
        count = sum(len(target) for _, target in group)
        if count == 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        for features, targets in group:
            if len(targets) == 0:
                continue
            features, targets = features.to(device), targets.to(device)
            dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
            with torch.autocast(device_type=device.type, dtype=dtype):
                summed_loss = F.cross_entropy(model(features), targets, reduction="sum")
            scaler.scale(summed_loss / count).backward()
            total_loss += float(summed_loss.detach())
        # Unscale once, after the complete effective batch, before clipping.
        scaler.unscale_(optimizer)
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()
        total_examples += count
    return total_loss / total_examples if total_examples else 0.0, scaler
