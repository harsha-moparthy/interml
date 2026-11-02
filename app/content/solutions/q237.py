"""Apply inverted dropout during training and preserve activations during evaluation."""

import numpy as np


class DropoutLayer:

    def __init__(self, p: float):
        if p < 0 or p >= 1:
            raise ValueError("Dropout rate must be between 0 and 1 (1-exclusive)")
        self.p = p
        self.mask = None

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not training:
            return x
        self.mask = np.random.binomial(1, 1 - self.p, x.shape)
        return x * self.mask / (1 - self.p)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.mask is None:
            raise ValueError("Forward pass must be called before backward pass")
        return grad * self.mask / (1 - self.p)
