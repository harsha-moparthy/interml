import torch


def q577(h, noise, sigma=0.1, logvar=None, training=True):
    """Implement additive or variational Gaussian bottleneck sampling with caller-supplied
    standard-normal noise. Without logvar return h+sigma*noise and zero KL; with logvar
    treat h as mu and return reparameterized z plus mean per-example KL to N(0,I).
    Evaluation returns the mean in either mode; noise and input shapes match.
    """
    if sigma < 0 or h.shape != noise.shape:
        raise ValueError("invalid noise")
    if logvar is None:
        return (h + sigma * noise if training else h, h.new_zeros(()))
    if logvar.shape != h.shape:
        raise ValueError("logvar shape")
    z = h + torch.exp(0.5 * logvar) * noise if training else h
    kl = 0.5 * (h.square() + logvar.exp() - 1 - logvar).sum(-1).mean()
    return (z, kl)
