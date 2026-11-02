def q107(x, weights, upstream):
    """Backpropagate y=x@weights+b. Shapes are x=(n,d), weights=(d,k), upstream=(n,k).
    Return gradients for x, weights, and bias.
    """
    return (upstream @ weights.T, x.T @ upstream, upstream.sum(axis=0))
