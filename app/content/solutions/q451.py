"""Learn a tabular action-value function using Q-learning updates."""

import numpy as np


def q_learning(
    num_states,
    num_actions,
    P,
    R,
    terminal_states,
    alpha,
    gamma,
    epsilon,
    num_episodes,
):
    Q = np.zeros((num_states, num_actions))
    for episode in range(num_episodes):
        state = np.random.choice(
            [s for s in range(num_states) if s not in set(terminal_states)]
        )
        while state not in terminal_states:
            if np.random.rand() < epsilon:
                action = np.random.randint(num_actions)
            else:
                action = np.argmax(Q[state])
            next_state = np.random.choice(num_states, p=P[state, action])
            reward = R[state, action]
            if next_state in terminal_states:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_state])
            Q[state, action] += alpha * (target - Q[state, action])
            state = next_state
    return Q
