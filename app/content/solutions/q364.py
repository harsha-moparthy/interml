import numpy as np


def q364(
    chosen_sum,
    rejected_sum,
    chosen_length,
    rejected_length,
    beta=2,
    margin=0.5,
):
    """Compute SimPO using length-normalized sequence log probabilities and a target
    preference margin.
    """
    score = (
        beta
        * (
            np.asarray(chosen_sum) / chosen_length
            - np.asarray(rejected_sum) / rejected_length
        )
        - margin
    )
    return float(np.logaddexp(0, -score).mean())
