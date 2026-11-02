import numpy as np


def q169(gradients, hessians, regularization=1, gamma=0):
    """For gradients and Hessians grouped by candidate leaf, compute optimal leaf weights
    and the minimized second-order XGBoost objective. Add gamma per leaf.
    """
    g = np.asarray(gradients, float)
    h = np.asarray(hessians, float)
    weights = -g / (h + regularization)
    objective = -0.5 * np.sum(g * g / (h + regularization)) + gamma * len(g)
    return (weights, float(objective))
