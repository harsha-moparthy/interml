import numpy as np


def q447(rewards, gamma):
    """Return every discounted reward-to-go for a finite episode, with terminal bootstrap
    zero.
    """
    out = np.empty(len(rewards))
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out
