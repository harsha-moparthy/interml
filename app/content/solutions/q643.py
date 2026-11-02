import numpy as np


def fit_gradient_boosting(x, y, *, stages=100, learning_rate=0.1):
    """Squared-loss GBDT with depth-one regression trees (stumps).

    Return intercept and (feature, threshold, left_update, right_update) stages.
    Each stage fits negative gradients y-prediction. Exhaustive thresholds keep
    the algorithm visible; deeper trees and histogram search are extensions.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (
        x.ndim != 2
        or not len(x)
        or y.shape != (len(x),)
        or stages < 1
        or learning_rate <= 0
    ):
        raise ValueError("invalid boosting inputs")
    intercept = float(y.mean())
    prediction = np.full(len(y), intercept)
    models = []
    for _ in range(stages):
        residual = y - prediction
        best, best_error = None, float(residual @ residual)
        for feature in range(x.shape[1]):
            values = np.unique(x[:, feature])
            for threshold in values[:-1] / 2 + values[1:] / 2:
                left = x[:, feature] <= threshold
                left_mean, right_mean = residual[left].mean(), residual[~left].mean()
                error = ((residual - np.where(left, left_mean, right_mean)) ** 2).sum()
                if error < best_error:
                    best_error = error
                    best = (
                        feature,
                        threshold,
                        learning_rate * left_mean,
                        learning_rate * right_mean,
                    )
        if best is None:
            break
        feature, threshold, left_update, right_update = best
        prediction += np.where(x[:, feature] <= threshold, left_update, right_update)
        models.append(best)
    return intercept, models


def predict_gradient_boosting(x, model):
    x = np.asarray(x)
    intercept, stages = model
    output = np.full(len(x), intercept)
    for feature, threshold, left, right in stages:
        output += np.where(x[:, feature] <= threshold, left, right)
    return output
