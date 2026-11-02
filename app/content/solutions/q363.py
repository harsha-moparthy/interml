import numpy as np


def q363(chosen_mean_logp, rejected_mean_logp, weight=0.1):
    """Compute ORPO from mean response-token log probabilities: chosen NLL plus weighted
    negative log-sigmoid of a log-odds ratio. Inputs must be strictly negative.
    """
    chosen, rejected = (
        np.asarray(chosen_mean_logp),
        np.asarray(rejected_mean_logp),
    )
    if np.any(chosen >= 0) or np.any(rejected >= 0):
        raise ValueError("log probabilities must be negative")
    log_odds = (
        chosen - np.log(-np.expm1(chosen)) - (rejected - np.log(-np.expm1(rejected)))
    )
    return float(np.mean(-chosen + weight * np.logaddexp(0, -log_odds)))
