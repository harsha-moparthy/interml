"""Evaluate a normal probability density at x for the supplied mean and standard deviation."""

import math


def normal_pdf(x, mean, std_dev):
    coefficient = 1 / (math.sqrt(2 * math.pi) * std_dev)
    exponent = math.exp(-((x - mean) ** 2) / (2 * std_dev**2))
    return round(coefficient * exponent, 5)
