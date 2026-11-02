import numpy as np


def q377(option_token_logps, length_normalize=False):
    """Select the multiple-choice option with largest summed continuation log probability;
    optionally divide each sum by its nonzero token count. Return index and scores.
    """
    if any((len(x) == 0 for x in option_token_logps)):
        raise ValueError("empty option")
    scores = np.array(
        [np.mean(x) if length_normalize else np.sum(x) for x in option_token_logps]
    )
    return (int(np.argmax(scores)), scores)
