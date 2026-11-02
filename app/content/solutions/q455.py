"""Update an action’s running mean reward from its previous count and mean."""


def incremental_mean(Q_prev, k, R):
    return Q_prev + 1 / k * (R - Q_prev)
