import numpy as np


class NaiveBayes:
    """Gaussian features or multinomial nonnegative counts, computed in log space."""

    def __init__(self, kind="gaussian", alpha=1.0, variance_floor=1e-9):
        if kind not in ("gaussian", "multinomial") or alpha <= 0 or variance_floor <= 0:
            raise ValueError("invalid model settings")
        self.kind, self.alpha, self.variance_floor = kind, alpha, variance_floor

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y)
        if (
            x.ndim != 2
            or not len(x)
            or y.shape != (len(x),)
            or not np.isfinite(x).all()
        ):
            raise ValueError("invalid training inputs")
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.log_prior_ = np.log(np.bincount(encoded) / len(y))
        if self.kind == "gaussian":
            self.mean_ = np.stack(
                [x[encoded == c].mean(0) for c in range(len(self.classes_))]
            )
            self.variance_ = np.stack(
                [x[encoded == c].var(0) for c in range(len(self.classes_))]
            ).clip(min=self.variance_floor)
        else:
            if np.any(x < 0):
                raise ValueError("multinomial inputs must be nonnegative")
            counts = (
                np.stack([x[encoded == c].sum(0) for c in range(len(self.classes_))])
                + self.alpha
            )
            self.log_likelihood_ = np.log(counts / counts.sum(1, keepdims=True))
        return self

    def predict(self, x):
        x = np.asarray(x, dtype=float)
        if self.kind == "gaussian":
            log_joint = self.log_prior_ - 0.5 * (
                np.log(2 * np.pi * self.variance_)[None]
                + (x[:, None] - self.mean_[None]) ** 2 / self.variance_[None]
            ).sum(-1)
        else:
            if np.any(x < 0):
                raise ValueError("multinomial inputs must be nonnegative")
            log_joint = x @ self.log_likelihood_.T + self.log_prior_
        return self.classes_[log_joint.argmax(-1)]
