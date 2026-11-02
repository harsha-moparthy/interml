import numpy as np


class LayerNorm:
    """Last-axis LayerNorm with analytical backward; one outstanding forward at a time."""

    def __init__(self, width, eps=1e-5):
        if width < 1 or eps <= 0:
            raise ValueError("positive width and epsilon required")
        self.weight = np.ones(width)
        self.bias = np.zeros(width)
        self.eps = eps
        self.cache = None

    def forward(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim < 1 or x.shape[-1] != len(self.weight):
            raise ValueError("last dimension must match width")
        centered = x - x.mean(-1, keepdims=True)
        inverse_std = 1 / np.sqrt(
            (centered * centered).mean(-1, keepdims=True) + self.eps
        )
        normalized = centered * inverse_std
        self.cache = normalized, inverse_std, self.weight.copy()
        return normalized * self.weight + self.bias

    def backward(self, upstream):
        if self.cache is None:
            raise RuntimeError("call forward before backward")
        normalized, inverse_std, weight = self.cache
        upstream = np.asarray(upstream, dtype=float)
        if upstream.shape != normalized.shape:
            raise ValueError("upstream shape must match output")
        gradient = upstream * weight
        dx = inverse_std * (
            gradient
            - gradient.mean(-1, keepdims=True)
            - normalized * (gradient * normalized).mean(-1, keepdims=True)
        )
        axes = tuple(range(upstream.ndim - 1))
        return dx, (upstream * normalized).sum(axis=axes), upstream.sum(axis=axes)
