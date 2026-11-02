def q429(
    parameter_count,
    batch,
    prompt_tokens,
    weight_bytes,
    peak_flops,
    bandwidth,
):
    """Compare ideal prefill and single-step decode dense-layer rooflines, assuming weights
    read once per batch. Ignore KV traffic and attention FLOPs. Return classifications
    and time lower bounds.
    """
    traffic = parameter_count * weight_bytes
    result = {}
    for name, tokens in (
        ("prefill", batch * prompt_tokens),
        ("decode", batch),
    ):
        compute = 2 * parameter_count * tokens / peak_flops
        memory = traffic / bandwidth
        result[name] = {
            "bound": "compute" if compute >= memory else "memory",
            "seconds": max(compute, memory),
        }
    return result
