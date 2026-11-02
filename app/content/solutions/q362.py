import numpy as np


def q362(policy_logp, reference_logp, beta=0.1):
    """Compute DPO implicit rewards beta*(log policy probability-log reference
    probability), omitting the prompt-dependent additive constant.
    """
    return beta * (np.asarray(policy_logp) - reference_logp)
