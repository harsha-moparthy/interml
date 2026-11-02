import numpy as np


def q352(targets, assistant_mask, ignore_index=-100):
    """Keep target tokens where an aligned assistant-response mask is True and replace all
    other target positions with ignore_index.
    """
    return np.where(assistant_mask, targets, ignore_index)
