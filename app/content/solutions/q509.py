import copy
import math
import torch
from torch.nn import functional as F


def q503(student, teacher_targets, temperature=0.1):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("invalid temperature")
    if student.ndim != 3 or student.shape[0] < 2 or student.shape[1] < 1:
        raise ValueError("invalid student views")
    if teacher_targets.shape != (2, *student.shape[1:]):
        raise ValueError("teacher shape mismatch")
    logp = F.log_softmax(student.float() / temperature, -1)
    losses = [
        -(teacher_targets[t].detach() * logp[v]).sum(-1).mean()
        for t in range(2)
        for v in range(len(student))
        if t != v
    ]
    return (torch.stack(losses).mean(), len(losses))


def q502(logits, center, temperature=0.04):
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("invalid temperature")
    return F.softmax(
        (logits.detach().float() - center.detach().float()) / temperature,
        dim=-1,
    )


def q509(
    student,
    teacher,
    views,
    optimizer,
    center,
    momentum=0.996,
    center_momentum=0.9,
    teacher_temp=0.04,
):
    """Run one small original-DINO step with supplied BN-free, dropout-free matching
    networks returning [B,K]. If teacher is None, deepcopy/freeze the student. The first
    two aligned views are global; the student sees all views. Optimize student,
    EMA-update teacher using the NEW student, update center from OLD raw teacher logits,
    and return (loss,teacher,new_center). This is the training core, not DINOv3
    pretraining; caller owns augmentation and checkpoints.
    """
    if len(views) < 2 or not 0 <= momentum <= 1 or (not 0 <= center_momentum <= 1):
        raise ValueError("invalid views/momentum")
    if teacher is None:
        teacher = copy.deepcopy(student).requires_grad_(False)
    if any((p.requires_grad for p in teacher.parameters())):
        raise ValueError("teacher must be frozen")
    student.train()
    teacher.eval()
    with torch.no_grad():
        old_logits = torch.stack([teacher(v) for v in views[:2]])
    logits = torch.stack([student(v) for v in views])
    loss, _ = q503(logits, q502(old_logits, center, teacher_temp))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        for target, source in zip(
            teacher.parameters(), student.parameters(), strict=True
        ):
            target.lerp_(source, 1 - momentum)
        for target, source in zip(teacher.buffers(), student.buffers(), strict=True):
            target.copy_(source)
        new_center = (
            center_momentum * center
            + (1 - center_momentum) * old_logits.mean((0, 1))[None]
        )
    return (loss.detach(), teacher, new_center)
