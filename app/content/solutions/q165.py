import numpy as np


def q165(x, y, test, trees=30, max_depth=5, seed=0):
    """Implement random-forest bagging and voting, using sklearn only for individual
    randomized CART trees. Each tree bootstraps rows and considers sqrt(d) features per
    split.
    """
    from sklearn.tree import DecisionTreeClassifier

    x, y = (np.asarray(x), np.asarray(y))
    rng = np.random.default_rng(seed)
    votes = []
    fitted = []
    for _ in range(trees):
        indices = rng.integers(len(x), size=len(x))
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            max_features="sqrt",
            random_state=int(rng.integers(2**31)),
        )
        model.fit(x[indices], y[indices])
        fitted.append(model)
        votes.append(model.predict(test))
    votes = np.stack(votes)
    classes = np.unique(y)
    result = classes[np.stack([(votes == c).sum(0) for c in classes]).argmax(0)]
    return (result, fitted)
