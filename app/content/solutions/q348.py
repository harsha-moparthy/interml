def q348(vocabulary, width, layers, context, ffn_width):
    """Count parameters for a tied-embedding GPT with learned positions, biased QKV/output
    projections, biased two-layer FFNs, two affine LayerNorms per block, and one final
    LayerNorm.
    """
    embeddings = vocabulary * width + context * width
    attention = 4 * width * width + 4 * width
    ffn = 2 * width * ffn_width + ffn_width + width
    norms = 4 * width
    return embeddings + layers * (attention + ffn + norms) + 2 * width
