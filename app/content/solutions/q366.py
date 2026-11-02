import numpy as np


def q366(
    policy_logp,
    old_logp,
    reference_logp,
    advantages,
    epsilon=0.2,
    beta=0.01,
):
    """Compute a token-mean GRPO objective to maximize from equally shaped arrays of
    sampled token log probabilities and broadcastable group-normalized advantages. Use
    clipped importance ratios and the nonnegative sampled k3 KL to the reference; all
    entries are valid tokens.
    """
    current = np.asarray(policy_logp, float)
    old = np.asarray(old_logp, float)
    reference = np.asarray(reference_logp, float)
    advantage = np.asarray(advantages, float)
    ratio = np.exp(current - old)
    surrogate = np.minimum(
        ratio * advantage,
        np.clip(ratio, 1 - epsilon, 1 + epsilon) * advantage,
    )
    difference = reference - current
    kl = np.expm1(difference) - difference
    return float(np.mean(surrogate - beta * kl))
