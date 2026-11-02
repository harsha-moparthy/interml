def q473(real_scores, fake_scores, gradient_penalty=0, penalty_weight=10):
    """Return the minimized Wasserstein critic objective
    mean(fake)-mean(real)+lambda*gradient_penalty. The caller supplies a differentiable
    scalar gradient penalty when desired.
    """
    return fake_scores.mean() - real_scores.mean() + penalty_weight * gradient_penalty
