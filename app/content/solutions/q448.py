import numpy as np


def q448(transition, reward, policy, gamma):
    """Solve the Bellman expectation linear system for a finite continuing MDP. P has shape
    (S,A,S), R is expected immediate reward (S,A), policy is (S,A), and 0<=gamma<1.
    """
    p = np.asarray(transition)
    pi = np.asarray(policy)
    r = np.asarray(reward)
    p_pi = np.einsum("sa,san->sn", pi, p)
    r_pi = np.sum(pi * r, axis=1)
    return np.linalg.solve(np.eye(len(p)) - gamma * p_pi, r_pi)
