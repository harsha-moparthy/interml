import numpy as np


def q456(reward_sums, counts, exploration=2):
    """Select an arm using mean_reward + sqrt(exploration*log(total_pulls)/count). Pull the
    first unvisited arm before computing confidence bounds.
    """
    counts = np.asarray(counts)
    sums = np.asarray(reward_sums)
    missing = np.flatnonzero(counts == 0)
    if len(missing):
        return int(missing[0])
    return int(
        np.argmax(sums / counts + np.sqrt(exploration * np.log(counts.sum()) / counts))
    )
