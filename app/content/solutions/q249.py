import numpy as np


def q249(
    parameters,
    gradient,
    first,
    second,
    step,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    decay=0.01,
):
    """Perform one AdamW update with bias correction and decoupled weight decay. Step is
    one-based; return parameters and moment states.
    """
    m = beta1 * np.asarray(first) + (1 - beta1) * gradient
    v = beta2 * np.asarray(second) + (1 - beta2) * np.asarray(gradient) ** 2
    # AdamW decays parameters separately from the adaptive gradient update.
    updated = np.asarray(parameters) * (1 - lr * decay) - lr * (
        m / (1 - beta1**step)
    ) / (np.sqrt(v / (1 - beta2**step)) + epsilon)
    return (updated, m, v)
