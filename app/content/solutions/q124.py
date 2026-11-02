"""Generate polynomial feature combinations up to the requested degree."""

import numpy as np
from itertools import combinations_with_replacement


def polynomial_features(X, degree):
    n_samples, n_features = X.shape
    combs = [
        c
        for d in range(degree + 1)
        for c in combinations_with_replacement(range(n_features), d)
    ]
    X_poly = np.empty((n_samples, len(combs)))
    for i, idx in enumerate(combs):
        X_poly[:, i] = 1 if len(idx) == 0 else np.prod(X[:, idx], axis=1)
    X_sorted = np.sort(X_poly, axis=1)
    return X_sorted
