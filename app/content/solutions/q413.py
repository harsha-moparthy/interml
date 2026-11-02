import numpy as np


def q413(weights, calibration, bits=4, damping=0.01):
    """Implement a small GPTQ-style columnwise quantizer for W(out,in) and calibration
    X(samples,in). Use a damped X.T@X Hessian, its inverse Cholesky factor, fixed
    per-row scales, and sequential error compensation; return reconstructed weights.
    """
    w = np.asarray(weights, float).copy()
    x = np.asarray(calibration, float)
    h = x.T @ x / len(x)
    h += np.eye(h.shape[0]) * max(damping * np.trace(h) / len(h), 1e-08)
    upper = np.linalg.cholesky(np.linalg.inv(h)).T
    top = 2 ** (bits - 1) - 1
    scale = np.maximum(np.max(np.abs(w), axis=1, keepdims=True) / top, 1e-12)
    output = np.zeros_like(w)
    for i in range(w.shape[1]):
        quant = np.clip(np.rint(w[:, i] / scale[:, 0]), -top, top) * scale[:, 0]
        output[:, i] = quant
        error = (w[:, i] - quant) / upper[i, i]
        w[:, i:] -= error[:, None] * upper[i, i:]
    return output
