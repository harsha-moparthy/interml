import numpy as np


def logistic_regression(x, y, *, l1=0.0, l2=0.0, learning_rate=0.1, iterations=1000):
    """Proximal gradient for mean logistic loss + l1*||w||1 + l2*||w||²/2.

    Features should be scaled. Bias is unpenalized. Return weight and bias.
    A fixed learning rate must respect the smooth objective's curvature.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or not len(x)
        or min(l1, l2) < 0
        or learning_rate <= 0
        or iterations < 1
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("invalid logistic regression inputs")
    weight, bias = np.zeros(x.shape[1]), 0.0
    for _ in range(iterations):
        logits = x @ weight + bias
        probabilities = np.exp(-np.logaddexp(0, -logits))
        error = probabilities - y
        weight -= learning_rate * (x.T @ error / len(x) + l2 * weight)
        # Proximal L1 step creates exact zeros, unlike a sign subgradient.
        weight = np.sign(weight) * np.maximum(np.abs(weight) - learning_rate * l1, 0)
        bias -= learning_rate * error.mean()
    return weight, bias
