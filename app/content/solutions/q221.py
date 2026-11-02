"""Compute the Swish activation x * sigmoid(beta * x)."""

import math


def swish(x: float) -> float:
    return x * (1 / (1 + math.exp(-x)))
