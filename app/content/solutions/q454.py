import numpy as np


def q454(values, transitions, alpha=0.1, gamma=0.99, lam=0.9):
    """Apply online accumulating-trace TD(lambda) to state values from
    (state,reward,next_state,terminal) transitions. Decay traces each step and clear
    them on true episode termination.
    """
    v = np.array(values, float, copy=True)
    trace = np.zeros_like(v)
    for s, r, ns, done in transitions:
        delta = r + (0 if done else gamma * v[ns]) - v[s]
        trace *= gamma * lam
        trace[s] += 1
        v += alpha * delta * trace
        if done:
            trace.fill(0)
    return v
