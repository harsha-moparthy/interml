import numpy as np


def q467(x, encoder_weight, encoder_bias, decoder_weight, decoder_bias):
    """Implement a one-hidden-layer dense autoencoder: ReLU encoding and sigmoid
    reconstruction. Use row-batch arrays and return latent representation and
    reconstruction.
    """
    latent = np.maximum(0, np.asarray(x) @ encoder_weight + encoder_bias)
    z = latent @ decoder_weight + decoder_bias
    reconstruction = np.exp(-np.logaddexp(0, -z))
    return (latent, reconstruction)
