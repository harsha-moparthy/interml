import numpy as np
from sklearn.base import clone


def q171(base_estimators, meta_estimator, x, y, test, folds=5):
    """Build a stacking classifier using out-of-fold positive-class probabilities to train
    a meta-classifier. Inputs have binary labels 0/1 and each fold must contain both
    classes. Refit base models on all training data for test predictions.
    """
    from sklearn.model_selection import StratifiedKFold

    x, y = (np.asarray(x), np.asarray(y))
    oof = np.zeros((len(y), len(base_estimators)))
    features = []
    splitter = StratifiedKFold(folds, shuffle=True, random_state=0)
    for j, base in enumerate(base_estimators):
        for train, valid in splitter.split(x, y):
            model = clone(base).fit(x[train], y[train])
            oof[valid, j] = model.predict_proba(x[valid])[:, 1]
        model = clone(base).fit(x, y)
        features.append(model.predict_proba(test)[:, 1])
    meta = clone(meta_estimator).fit(oof, y)
    return meta.predict(np.stack(features, axis=1))
