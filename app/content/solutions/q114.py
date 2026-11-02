def q114(contributions):
    """Accumulate gradient contributions from shared graph paths. Input is a list of
    (node_id, scalar_gradient); return sums per node.
    """
    result = {}
    for node, gradient in contributions:
        result[node] = result.get(node, 0.0) + gradient
    return result
