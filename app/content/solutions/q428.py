def q428(batch, heads, query_length, key_length, head_dim, bytes_per_element=2):
    """Estimate dense attention matmul FLOPs (QK and PV only) and minimal input/output
    bytes. Also return additional bytes for writing and rereading one materialized score
    matrix; omit softmax work and repeated loads.
    """
    scale = batch * heads
    return {
        "matmul_flops": 4 * scale * query_length * key_length * head_dim,
        "input_output_bytes": scale
        * (2 * query_length + 2 * key_length)
        * head_dim
        * bytes_per_element,
        "score_roundtrip_bytes": 2
        * scale
        * query_length
        * key_length
        * bytes_per_element,
    }
