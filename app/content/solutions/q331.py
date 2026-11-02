import numpy as np


def q331(x, positions, base=10000):
    """Apply rotary position embeddings to adjacent coordinate pairs of (...,L,D) arrays. D
    must be even; positions is length L. Preserve pairwise Euclidean norms.
    """
    x = np.asarray(x, float)
    head_dim = x.shape[-1]
    if head_dim % 2:
        raise ValueError("even head dimension required")
    angle = np.asarray(positions)[:, None] * base ** (
        -np.arange(0, head_dim, 2) / head_dim
    )
    cosine, sine = (np.cos(angle), np.sin(angle))
    # Each adjacent coordinate pair is a 2-D rotation, preserving its squared norm.
    even, odd = (x[..., 0::2], x[..., 1::2])
    output = np.empty_like(x)
    output[..., 0::2] = even * cosine - odd * sine
    output[..., 1::2] = even * sine + odd * cosine
    return output
