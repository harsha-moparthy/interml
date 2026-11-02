from torch.nn import functional as F


def q514(image, residual, task="sr"):
    """Add a predicted residual to a bicubic-upsampled SR base or an unchanged enhancement
    base. Both images are in the same image-space range. Return unclipped predictions;
    reject shape mismatch or unknown task.
    """
    if task == "sr":
        base = F.interpolate(
            image,
            size=residual.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
    elif task == "enhancement":
        base = image
    else:
        raise ValueError("unknown task")
    if base.shape != residual.shape:
        raise ValueError("shape mismatch")
    return base + residual
