import torch.nn.functional as F


def dpo_loss(chosen, rejected, reference_chosen, reference_rejected, beta=0.1):
    """Mean DPO loss on sequence log-probabilities; padding/prompt masking is upstream.

    Reference scores are fixed. Sum completion-token log probabilities before
    calling; sequence averaging changes the objective. Cost is linear in batch size.
    """
    if beta <= 0 or not (
        chosen.shape
        == rejected.shape
        == reference_chosen.shape
        == reference_rejected.shape
    ):
        raise ValueError("positive beta and matching shapes required")
    policy_ratio = chosen - rejected
    reference_ratio = reference_chosen.detach() - reference_rejected.detach()
    return F.softplus(-beta * (policy_ratio - reference_ratio)).mean()
