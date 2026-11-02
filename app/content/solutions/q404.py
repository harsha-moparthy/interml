def q404(
    batch,
    tokens,
    layers,
    query_heads,
    kv_heads,
    head_dim,
    latent_dim,
    rope_dim,
    bytes_per_element=2,
):
    """Estimate KV bytes for MHA, GQA/MQA, and MLA. MLA stores one compressed latent plus
    one shared positional key per token per layer; ignore allocator overhead.
    """
    base = batch * tokens * layers * bytes_per_element
    return {
        "MHA": base * 2 * query_heads * head_dim,
        "GQA_or_MQA": base * 2 * kv_heads * head_dim,
        "MLA": base * (latent_dim + rope_dim),
    }
