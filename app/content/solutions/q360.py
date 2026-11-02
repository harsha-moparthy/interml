import numpy as np


def q360(chosen_reward, rejected_reward):
    """Compute Bradley-Terry reward-model loss, the mean negative log probability that each
    chosen response beats its paired rejected response.
    """
    difference = np.asarray(chosen_reward) - rejected_reward
    return float(np.logaddexp(0, -difference).mean())
