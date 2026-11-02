import torch
from torch.nn import functional as F


def q273(decoder, skip):
    """Implement a U-Net skip merge: bilinearly resize decoder features to the encoder
    skip's spatial size and concatenate along channels.
    """
    up = F.interpolate(
        decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False
    )
    return torch.cat([skip, up], dim=1)
