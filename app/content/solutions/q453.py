import numpy as np


def q447(rewards, gamma):
    out = np.empty(len(rewards))
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def q453(episodes, gamma=0.99):
    """Estimate first-visit Monte Carlo values from episodes of (state,reward_after_state)
    pairs. Average returns only at each state's first occurrence in each episode.
    """
    totals = {}
    counts = {}
    for episode in episodes:
        returns = q447([r for _, r in episode], gamma)
        seen = set()
        for (state, _), value in zip(episode, returns):
            if state in seen:
                continue
            seen.add(state)
            totals[state] = totals.get(state, 0) + value
            counts[state] = counts.get(state, 0) + 1
    return {s: totals[s] / counts[s] for s in totals}
