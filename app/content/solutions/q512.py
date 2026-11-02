import torch
from torch.nn import functional as F


def q512(images, crop=32, scale=4, seed=0):
    """Create paired HR/LR batches by independently cropping each image, then bicubic
    downsampling with antialiasing and align_corners=False. Return (HR,LR,coordinates).
    No output clipping; crop must fit and be divisible by positive integer scale.
    """
    if scale < 1 or crop < 1 or crop % scale or (crop > min(images.shape[-2:])):
        raise ValueError("invalid crop/scale")
    rng = torch.Generator().manual_seed(seed)
    crops = []
    coords = []
    for image in images:
        y = int(torch.randint(image.shape[-2] - crop + 1, (), generator=rng))
        x = int(torch.randint(image.shape[-1] - crop + 1, (), generator=rng))
        crops.append(image[:, y : y + crop, x : x + crop])
        coords.append((y, x))
    hr = torch.stack(crops)
    lr = F.interpolate(
        hr,
        size=(crop // scale, crop // scale),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return (hr, lr, coords)
