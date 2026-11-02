from torch.nn import functional as F


def q334(x, w1, b1, w2, b2):
    """Implement a GPT feed-forward block Linear-GELU-Linear. Matrix layout is
    input-by-output; apply exact GELU.
    """
    return F.gelu(x @ w1 + b1, approximate="none") @ w2 + b2
