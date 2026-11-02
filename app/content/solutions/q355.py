import numpy as np


def q355(x, quantized_weight, scale, a, b, alpha):
    """Implement a simple quantized-LoRA forward reference: dequantize a signed integer
    base matrix using row scales, keep it frozen, and add a floating-point low-rank
    update. This illustrates QLoRA composition, without NF4/double quantization.
    """
    weight = np.asarray(quantized_weight, float) * np.asarray(scale)[:, None]
    return x @ weight.T + alpha / a.shape[0] * (x @ a.T) @ b.T
