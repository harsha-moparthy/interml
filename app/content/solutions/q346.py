import numpy as np


def q346(queries, keys, values, decay, beta):
    """Implement a recurrent gated DeltaNet core for one sequence. Decay the (Dk,Dv) state,
    then add beta*k outer (v-k@state); emit q@state at each step. Inputs provide
    already-normalized keys and scalar gates.
    """
    q, k, v = (
        np.asarray(queries, float),
        np.asarray(keys, float),
        np.asarray(values, float),
    )
    state = np.zeros((k.shape[1], v.shape[1]))
    out = []
    for i in range(len(q)):
        state *= decay[i]
        residual = v[i] - k[i] @ state
        state += beta[i] * np.outer(k[i], residual)
        out.append(q[i] @ state)
    return (np.stack(out), state)
