"""Compute the probability of exactly k successes in n Bernoulli trials."""

import math


def binomial_probability(n, k, p):
    binomial_coeff = math.comb(n, k)
    probability = binomial_coeff * p**k * (1 - p) ** (n - k)
    return round(probability, 5)
