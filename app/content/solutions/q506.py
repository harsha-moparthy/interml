import torch


def q506(embeddings, probabilities):
    """Return mean population feature standard deviation, mean example entropy, entropy of
    batch-mean probabilities, and argmax prototype counts. Inputs are nonempty [B,D]
    embeddings and normalized nonnegative [B,K] probabilities. Compare uniform targets
    with single-prototype collapse.
    """

    def entropy(p):
        return -(p * p.clamp_min(1e-30).log()).sum(-1)

    return {
        "feature_std": embeddings.std(0, correction=0).mean(),
        "example_entropy": entropy(probabilities).mean(),
        "marginal_entropy": entropy(probabilities.mean(0)),
        "usage": torch.bincount(
            probabilities.argmax(-1), minlength=probabilities.shape[-1]
        ),
    }
