"""Compute the Poisson probability of exactly k events for the supplied rate."""

import math


def poisson_probability(k, lam):
    probability = lam**k * math.exp(-lam) / math.factorial(k)
    return round(probability, 5)
