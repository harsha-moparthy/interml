def q424(compute_budget, a, b, alpha, beta, flops_per_parameter_token=6):
    """Minimize loss a*N^-alpha+b*D^-beta under compute=k*N*D. Return optimal positive
    parameter count N and token count D for positive coefficients and exponents.
    """
    values = (compute_budget, a, b, alpha, beta, flops_per_parameter_token)
    if min(values) <= 0:
        raise ValueError("positive inputs required")
    product = compute_budget / flops_per_parameter_token
    n = (alpha * a / (beta * b) * product**beta) ** (1 / (alpha + beta))
    return (n, product / n)
