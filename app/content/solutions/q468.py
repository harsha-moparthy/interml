import torch


def q468(mu, log_variance):
    """Sample a differentiable diagonal Gaussian latent with the reparameterization trick.
    Return z and the batch-mean KL to a standard-normal prior, summing over latent
    dimensions.
    """
    z = mu + torch.exp(0.5 * log_variance) * torch.randn_like(mu)
    kl = 0.5 * (mu.square() + log_variance.exp() - 1 - log_variance).sum(-1).mean()
    return (z, kl)
