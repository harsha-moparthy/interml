def q430(elements, element_bytes, pointwise_stages):
    """Estimate global-memory traffic for a chain of shape-preserving unary pointwise
    operations. Separate kernels each read and write the array; one fused kernel reads
    once and writes once.
    """
    if pointwise_stages < 1:
        raise ValueError("at least one stage required")
    fused = 2 * elements * element_bytes
    separate = fused * pointwise_stages
    return {
        "separate_bytes": separate,
        "fused_bytes": fused,
        "saved_bytes": separate - fused,
        "reduction_fraction": 1 - 1 / pointwise_stages,
    }
