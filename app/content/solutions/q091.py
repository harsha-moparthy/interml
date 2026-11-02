def q091(successes, trials, alpha, beta):
    """Compute the interior MAP estimate of a Bernoulli probability under a Beta prior.
    Require posterior alpha and beta greater than one.
    """
    a = alpha + successes
    b = beta + trials - successes
    if a <= 1 or b <= 1:
        raise ValueError("posterior mode is not interior")
    return (a - 1) / (a + b - 2)
