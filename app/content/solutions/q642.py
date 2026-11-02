import numpy as np


class DecisionTree:
    """Small CART classifier with Gini splits and optional per-node feature sampling.

    Finite continuous features only; no missing-value routing or pruning.
    Exhaustive candidate evaluation prioritizes clarity over training throughput.
    """

    def __init__(self, max_depth=5, min_leaf=1, max_features=None, seed=0):
        if max_depth < 0 or min_leaf < 1:
            raise ValueError("invalid tree settings")
        self.max_depth, self.min_leaf = max_depth, min_leaf
        self.max_features, self.seed = max_features, seed

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y)
        if (
            x.ndim != 2
            or not len(x)
            or y.shape != (len(x),)
            or not np.isfinite(x).all()
        ):
            raise ValueError("expected finite nonempty features and matching labels")
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_features_ = x.shape[1]
        rng = np.random.default_rng(self.seed)
        count = self.n_features_ if self.max_features is None else self.max_features
        if not 1 <= count <= self.n_features_:
            raise ValueError("invalid feature count")

        def impurity(labels):
            probabilities = np.bincount(labels, minlength=len(self.classes_)) / len(
                labels
            )
            return 1 - probabilities @ probabilities

        def build(rows, depth):
            labels = encoded[rows]
            prediction = int(np.bincount(labels, minlength=len(self.classes_)).argmax())
            parent_impurity = impurity(labels)
            if (
                depth >= self.max_depth
                or parent_impurity == 0
                or len(rows) < 2 * self.min_leaf
            ):
                return prediction
            best, best_gain = None, 0.0
            for feature in sorted(rng.choice(self.n_features_, count, replace=False)):
                values = np.unique(x[rows, feature])
                for threshold in values[:-1] / 2 + values[1:] / 2:
                    left = x[rows, feature] <= threshold
                    if min(left.sum(), (~left).sum()) < self.min_leaf:
                        continue
                    gain = parent_impurity - (
                        left.mean() * impurity(labels[left])
                        + (~left).mean() * impurity(labels[~left])
                    )
                    if gain > best_gain:
                        best, best_gain = (feature, threshold, left), gain
            if best is None:
                return prediction
            feature, threshold, left = best
            return (
                feature,
                threshold,
                build(rows[left], depth + 1),
                build(rows[~left], depth + 1),
            )

        self.tree_ = build(np.arange(len(x)), 0)
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features_ or not np.isfinite(x).all():
            raise ValueError("invalid prediction features")

        def predict_row(row):
            node = self.tree_
            while isinstance(node, tuple):
                feature, threshold, left, right = node
                node = left if row[feature] <= threshold else right
            return self.classes_[node]

        return np.asarray([predict_row(row) for row in x])


class RandomForest:
    """Bootstrap examples per tree and sample sqrt(features) at EACH split."""

    def __init__(self, trees=50, max_depth=8, seed=0):
        if trees < 1:
            raise ValueError("at least one tree required")
        self.trees, self.max_depth, self.seed = trees, max_depth, seed

    def fit(self, x, y):
        x, y = np.asarray(x), np.asarray(y)
        if x.ndim != 2 or not len(x) or y.shape != (len(x),):
            raise ValueError("invalid training data")
        self.classes_ = np.unique(y)
        rng = np.random.default_rng(self.seed)
        self.models_ = []
        for _ in range(self.trees):
            rows = rng.integers(len(x), size=len(x))
            tree = DecisionTree(
                self.max_depth,
                max_features=max(1, int(np.sqrt(x.shape[1]))),
                seed=int(rng.integers(2**31)),
            )
            self.models_.append(tree.fit(x[rows], y[rows]))
        return self

    def predict(self, x):
        predictions = np.stack([tree.predict(x) for tree in self.models_], axis=1)
        votes = (predictions[..., None] == self.classes_).sum(axis=1)
        return self.classes_[votes.argmax(axis=1)]
