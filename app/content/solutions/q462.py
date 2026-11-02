import torch


def q462(new_logp, old_logp, advantages, epsilon=0.2):
    """Return Torch PPO clipped policy loss, clip fraction, and sampled k3 KL estimate.
    Treat old policy log probabilities and advantages as fixed targets.
    """
    difference = new_logp - old_logp.detach()
    ratio = difference.exp()
    a = advantages.detach()
    loss = -torch.minimum(ratio * a, ratio.clamp(1 - epsilon, 1 + epsilon) * a).mean()
    return (
        loss,
        ((ratio - 1).abs() > epsilon).float().mean().detach(),
        (ratio - 1 - difference).mean().detach(),
    )
