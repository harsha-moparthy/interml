import numpy as np


def q102(function, x, epsilon=1e-05):
    """Compute a numerical Jacobian of a vector-valued function at a vector x using central
    differences. Return shape (output_size,input_size).
    """
    x = np.asarray(x, float)
    columns = []
    for i in range(len(x)):
        shift = np.zeros_like(x)
        shift[i] = epsilon
        columns.append(
            (np.asarray(function(x + shift)) - function(x - shift)) / (2 * epsilon)
        )
    return np.stack(columns, axis=-1)
