import numpy as np


def q461(
    rewards,
    values,
    next_values,
    terminated,
    episode_end,
    gamma=0.99,
    lam=0.95,
):
    """Compute GAE for a flat rollout. True termination removes value bootstrap; either
    termination or truncation ends the advantage recursion. next_values must contain
    final-observation values for truncated transitions.
    """
    r = np.asarray(rewards)
    v = np.asarray(values)
    delta = r + gamma * ~np.asarray(terminated, bool) * np.asarray(next_values) - v
    advantages = np.zeros_like(delta, dtype=float)
    carry = 0.0
    # Truncation stops the trace but retains the final-state value bootstrap.
    for t in range(len(r) - 1, -1, -1):
        carry = delta[t] + gamma * lam * (not episode_end[t]) * carry
        advantages[t] = carry
    return (advantages, advantages + v)
