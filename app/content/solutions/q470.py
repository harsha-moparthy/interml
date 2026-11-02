from torch import nn
from torch.nn import functional as F


def q470(input_dim, hidden_dim, latent_dim, code_count, beta=0.25):
    """Build a trainable dense VQ-VAE Torch module. Forward returns reconstruction,
    nearest-code indices, and MSE reconstruction plus codebook and commitment loss.
    Latents use straight-through gradients.
    """

    class VQVAE(nn.Module):

        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.codes = nn.Embedding(code_count, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            distance = (z[:, None, :] - self.codes.weight[None, :, :]).square().sum(-1)
            index = distance.argmin(-1)
            quantized = self.codes(index)
            penalty = F.mse_loss(quantized, z.detach()) + beta * F.mse_loss(
                z, quantized.detach()
            )
            straight = z + (quantized - z).detach()
            reconstruction = self.decoder(straight)
            return (
                reconstruction,
                index,
                F.mse_loss(reconstruction, x) + penalty,
            )

    return VQVAE()
