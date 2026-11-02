def q557(base, adapters, weights):
    """Merge weighted LoRA deltas into an independent base-weight tensor. Each adapter is
    (A[r,in],B[out,r],alpha); compute sum coefficient*(alpha/r)*B@A. Different ranks are
    supported. Do not multiply separately averaged A/B matrices.
    """
    if len(adapters) != len(weights):
        raise ValueError("weights length")
    merged = base.clone()
    for (a, b, alpha), coefficient in zip(adapters, weights):
        if a.shape[0] < 1 or b.shape[1] != a.shape[0]:
            raise ValueError("rank mismatch")
        delta = b @ a
        if delta.shape != base.shape:
            raise ValueError("base shape mismatch")
        merged = merged + coefficient * (alpha / a.shape[0]) * delta
    # Sum full low-rank products; separately averaging A and B creates cross terms.
    return merged
