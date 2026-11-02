import numpy as np


def q361(
    chosen_logp,
    rejected_logp,
    reference_chosen,
    reference_rejected,
    beta=0.1,
):
    """Compute the mean DPO loss from sequence log probabilities under a policy and
    reference policy. Use stable softplus of the negative preference margin.
    """
    margin = beta * (
        np.asarray(chosen_logp)
        - rejected_logp
        - (np.asarray(reference_chosen) - reference_rejected)
    )
    # logaddexp(0, -margin) is softplus(-margin) without overflow.
    return float(np.logaddexp(0, -margin).mean())
