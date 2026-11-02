from torch.nn import functional as F


def q576(logits, labels, reconstruction, target, weight=1.0):
    """Combine mean intent cross-entropy and mean feature reconstruction MSE against a
    detached target. Return total, classification loss, reconstruction loss. Zero
    reconstruction weight removes its gradient influence. The caller supplies the
    encoder/classifier/decoder outputs.
    """
    if weight < 0:
        raise ValueError("negative loss weight")
    classification = F.cross_entropy(logits, labels)
    reconstruction_loss = F.mse_loss(reconstruction, target.detach())
    return (
        classification + weight * reconstruction_loss,
        classification,
        reconstruction_loss,
    )
