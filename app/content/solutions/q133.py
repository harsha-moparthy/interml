import numpy as np
from sklearn.base import clone


def q133(specifications, x, test):
    """Fit column-specific transformers and concatenate their transformed outputs.
    Specifications is a list of (column_indices, transformer). This reference assumes
    dense outputs.
    """
    train_parts = []
    test_parts = []
    fitted = []
    for columns, transformer in specifications:
        t = clone(transformer)
        train_parts.append(t.fit_transform(np.asarray(x)[:, columns]))
        test_parts.append(t.transform(np.asarray(test)[:, columns]))
        fitted.append(t)
    return (
        np.concatenate(train_parts, axis=1),
        np.concatenate(test_parts, axis=1),
        fitted,
    )
