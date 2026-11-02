def q354(x, weight, a, b, alpha):
    """Compute a LoRA-adapted linear layer with weight (out,in), A (rank,in), B (out,rank),
    and scaling alpha/rank. Return xW.T+alpha/rank*xA.TB.T.
    """
    return x @ weight.T + alpha / a.shape[0] * (x @ a.T) @ b.T
