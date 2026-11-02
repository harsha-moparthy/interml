import numpy as np


def q343(router_probabilities, selected_experts):
    """Compute Switch-style MoE load balancing loss for top-1 routing: number_of_experts
    times the dot product of mean router probabilities and token assignment fractions.
    """
    p = np.asarray(router_probabilities, float)
    experts = p.shape[1]
    fractions = np.bincount(selected_experts, minlength=experts) / len(p)
    # Balance both routing frequency and probability mass; hard counts alone have no gradient.
    return float(experts * np.dot(p.mean(0), fractions))
