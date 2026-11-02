from sklearn.base import clone


def q132(transformers, estimator, x, y, test):
    """Fit a chain of sklearn-compatible transformers on training data only, fit a cloned
    estimator, and predict test data. Return predictions and fitted components.
    """
    a, b = (x, test)
    fitted = []
    for transformer in transformers:
        current = clone(transformer)
        a = current.fit_transform(a, y)
        b = current.transform(b)
        fitted.append(current)
    model = clone(estimator).fit(a, y)
    return (model.predict(b), fitted, model)
