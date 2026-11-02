from torch.nn import functional as F


def q466(
    policy_logits,
    value_predictions,
    policy_targets,
    value_targets,
    parameters=(),
    l2=0,
):
    """Compute AlphaZero cross-entropy policy loss plus value MSE and optional unnormalized
    L2 parameter penalty. Return a differentiable scalar Torch tensor.
    """
    policy = -(policy_targets * F.log_softmax(policy_logits, dim=-1)).sum(-1).mean()
    value = F.mse_loss(value_predictions.reshape(-1), value_targets.reshape(-1))
    return (
        policy
        + value
        + l2 * sum((p.square().sum() for p in parameters), policy.new_zeros(()))
    )
