import numpy as np
from sklearn.base import clone


def q164(estimator, x, y, test, models=20, seed=0):
    """Implement bootstrap bagging around a supplied sklearn-compatible classifier. Fit
    independent clones and majority-vote predictions, breaking ties by sorted label
    order.
    """
    x, y = (np.asarray(x), np.asarray(y))
    rng = np.random.default_rng(seed)
    predictions = []
    for _ in range(models):
        indices = rng.integers(len(x), size=len(x))
        predictions.append(clone(estimator).fit(x[indices], y[indices]).predict(test))
    classes = np.unique(y)
    votes = np.stack(predictions)
    return classes[np.stack([(votes == c).sum(0) for c in classes]).argmax(0)]
