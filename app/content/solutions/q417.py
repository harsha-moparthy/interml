import numpy as np


def q417(activations, weights, alpha=0.5, epsilon=1e-08):
    """Compute SmoothQuant migration scales for X(samples,in) and W(in,out), returning X/s,
    s*W, and s. Their floating-point product must equal X@W.
    """
    x = np.asarray(activations, float)
    w = np.asarray(weights, float)
    a = np.maximum(np.abs(x).max(axis=0), epsilon)
    b = np.maximum(np.abs(w).max(axis=1), epsilon)
    scale = a**alpha / b ** (1 - alpha)
    return (x / scale, scale[:, None] * w, scale)
