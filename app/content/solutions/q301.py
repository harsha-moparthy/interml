import numpy as np


def q301(
    word,
    context,
    bias_word,
    bias_context,
    count,
    xmax=100,
    alpha=0.75,
    lr=0.01,
):
    """Perform one GloVe pair update for a positive co-occurrence count. Minimize
    f(count)*(word dot context+biases-log(count))^2 and use simultaneous gradients.
    """
    if count <= 0:
        raise ValueError("positive count required")
    word, context = (np.asarray(word, float), np.asarray(context, float))
    weight = min((count / xmax) ** alpha, 1)
    error = word @ context + bias_word + bias_context - np.log(count)
    gradient = 2 * weight * error
    return (
        word - lr * gradient * context,
        context - lr * gradient * word,
        bias_word - lr * gradient,
        bias_context - lr * gradient,
        float(weight * error * error),
    )
