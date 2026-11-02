import numpy as np


def q347(state, key, value, beta):
    """Apply one delta-rule associative-memory update S'=S+beta*k*(v-k@S). Key is a vector;
    value has the output-memory dimension.
    """
    s, k, v = (
        np.asarray(state, float),
        np.asarray(key, float),
        np.asarray(value, float),
    )
    return s + beta * np.outer(k, v - k @ s)
