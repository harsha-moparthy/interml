"""Evaluate sigmoid and return the exercise-required four-decimal result."""

import math


def sigmoid(z: float) -> float:
    result = 1 / (1 + math.exp(-z))
    return round(result, 4)
