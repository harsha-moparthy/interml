import numpy as np


def q368(policy_logp, reference_logp):
    """Compute the nonnegative k3 sampled KL estimator
    exp(log_ref-log_policy)-(log_ref-log_policy)-1 for samples drawn from the policy.
    """
    difference = np.asarray(reference_logp) - policy_logp
    return np.expm1(difference) - difference
