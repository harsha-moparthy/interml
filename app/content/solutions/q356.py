def q356(weight, a, b, alpha):
    """Merge LoRA adapter matrices into a base (out,in) weight for inference. Return a new
    matrix, leaving the originals unchanged.
    """
    return weight + alpha / a.shape[0] * (b @ a)
