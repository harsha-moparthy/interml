import numpy as np


def q491(user_vector, item_vector, rating, lr, reg):
    """Perform one simultaneous SGD update for half squared rating error plus half reg
    times both squared vector norms. Use pre-update vectors in both gradients.
    """
    u, v = (np.array(user_vector, float), np.array(item_vector, float))
    error = u @ v - rating
    return (u - lr * (error * v + reg * u), v - lr * (error * u + reg * v))
