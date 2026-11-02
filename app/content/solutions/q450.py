import numpy as np


def q450(transition, reward, gamma=0.99, max_iterations=1000):
    """Implement exact policy iteration for a finite discounted MDP with P(S,A,S) and
    expected R(S,A). Initialize action zero; evaluate with a linear solve and improve
    greedily until stable.
    """
    p = np.asarray(transition)
    r = np.asarray(reward)
    states = np.arange(len(p))
    policy = np.zeros(len(p), int)
    for _ in range(max_iterations):
        value = np.linalg.solve(
            np.eye(len(p)) - gamma * p[states, policy], r[states, policy]
        )
        new = np.argmax(r + gamma * np.einsum("san,n->sa", p, value), axis=1)
        if np.array_equal(new, policy):
            return (policy, value)
        policy = new
    raise RuntimeError("policy iteration did not converge")
