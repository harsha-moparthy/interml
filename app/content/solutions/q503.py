import math
import torch
from torch.nn import functional as F


def q503(student, teacher_targets, temperature=0.1):
    """Compute original-DINO cross-view loss. Student logits are [V,B,K], V>=2; detached
    teacher probabilities are [2,B,K]. Exclude student/teacher pairs (0,0) and (1,1),
    average over examples and remaining pairs, and return (loss,pair_count).
    """
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
