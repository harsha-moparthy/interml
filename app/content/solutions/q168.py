import numpy as np


def q168(x, y, prediction, learning_rate=0.1, max_depth=2):
    """Perform one squared-error gradient boosting round by fitting a shallow regression
    tree to residuals. Return updated training predictions and the fitted tree.
    """
    from sklearn.tree import DecisionTreeRegressor

    residual = np.asarray(y) - prediction
    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=0).fit(x, residual)
    return (prediction + learning_rate * tree.predict(x), tree)
