import numpy as np


def q247(parameters, gradient, square_average, lr=0.001, decay=0.9, epsilon=1e-08):
    """Perform one uncentered RMSProp update, returning new parameters and the new running
    squared-gradient average.
    """
    average = (
        decay * np.asarray(square_average) + (1 - decay) * np.asarray(gradient) ** 2
    )
    return (
        parameters - lr * gradient / (np.sqrt(average) + epsilon),
        average,
    )
