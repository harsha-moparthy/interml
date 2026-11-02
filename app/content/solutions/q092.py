def q092(alpha, beta, successes, failures):
    """Update a Beta prior after Bernoulli observations. Return posterior shape parameters,
    mean, and variance.
    """
    a, b = (alpha + successes, beta + failures)
    return (a, b, a / (a + b), a * b / ((a + b) ** 2 * (a + b + 1)))
