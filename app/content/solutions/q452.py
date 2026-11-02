import numpy as np


def q452(q, transitions, alpha=0.1, gamma=0.99):
    """Apply sequential SARSA updates to a copy of a Q table from tuples
    (state,action,reward,next_state,next_action,terminal). A terminal transition has no
    bootstrap.
    """
    q = np.array(q, float, copy=True)
    for s, a, r, ns, na, done in transitions:
        target = r if done else r + gamma * q[ns, na]
        q[s, a] += alpha * (target - q[s, a])
    return q
