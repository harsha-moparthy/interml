def q543(x0, x, weight, bias, u=None, v=None):
    """Compute a DCN-V2 cross layer x0*(x@W.T+b)+x. Either supply dense W or supply factors
    U[D,r],V[D,r] with weight=None, representing W=U@V.T. Return [B,D]; a zero
    projection/bias recovers x.
    """
    if weight is None:
        projection = x @ v @ u.T
    else:
        projection = x @ weight.T
    return x0 * (projection + bias) + x
