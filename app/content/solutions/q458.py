import torch
from torch.nn import functional as F


def q458(
    online,
    target,
    optimizer,
    states,
    actions,
    rewards,
    next_states,
    terminated,
    gamma=0.99,
):
    """Perform one Double-DQN training step using supplied Torch networks and optimizer.
    Online selects the next action, target evaluates it; true terminal transitions do
    not bootstrap. Return detached Huber loss.
    """
    prediction = online(states).gather(1, actions.long()[:, None]).squeeze(1)
    with torch.no_grad():
        next_action = online(next_states).argmax(1, keepdim=True)
        next_value = target(next_states).gather(1, next_action).squeeze(1)
        labels = rewards + gamma * (~terminated.bool()).to(rewards.dtype) * next_value
    loss = F.smooth_l1_loss(prediction, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())
