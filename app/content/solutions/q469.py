from torch.nn import functional as F


def q469(reconstruction_logits, targets, mu, log_variance, beta=1):
    """Compute the negative Bernoulli VAE ELBO: summed per-example reconstruction BCE plus
    beta times latent KL, then average over batch.
    """
    reconstruction = (
        F.binary_cross_entropy_with_logits(
            reconstruction_logits, targets, reduction="none"
        )
        .flatten(1)
        .sum(1)
    )
    kl = 0.5 * (mu.square() + log_variance.exp() - 1 - log_variance).sum(-1)
    return (reconstruction + beta * kl).mean()
