def q471(latents, codebook):
    """Quantize last-axis Torch latent vectors to nearest codebook rows. Return
    straight-through quantized values and code indices; codebook learning requires a
    separate codebook loss.
    """
    flat = latents.reshape(-1, latents.shape[-1])
    distances = (
        flat.square().sum(1, keepdim=True)
        + codebook.square().sum(1)
        - 2 * flat @ codebook.T
    )
    index = distances.argmin(1)
    quantized = codebook[index].reshape_as(latents)
    return (
        latents + (quantized - latents).detach(),
        index.reshape(latents.shape[:-1]),
    )
