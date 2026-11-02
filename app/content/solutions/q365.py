import numpy as np


def q365(task_rewards, policy_logp, reference_logp, beta, target_kl, horizon=100):
    """Apply sampled log-ratio KL reward shaping and update a nonnegative KL controller.
    Return shaped rewards and beta multiplied by exp(clipped relative KL error/horizon).
    """
    kl = np.asarray(policy_logp) - reference_logp
    shaped = np.asarray(task_rewards) - beta * kl
    error = np.clip(kl.mean() / target_kl - 1, -0.2, 0.2)
    return (shaped, float(beta * np.exp(error / horizon)))
