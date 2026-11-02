from torch import nn
from torch.nn import functional as F


def q514(image, residual, task="sr"):
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


def q522(encoder, channels, patch_size=16, sr_scale=4):
    """Return a compact two-head restoration model using a supplied trainable/adapted
    encoder that emits PATCH-ONLY tokens [B,H/p*W/p,C]. Reshape tokens, explicitly
    interpolate to input resolution, predict an SR residual with PixelShuffle and an
    enhancement residual with a convolution. No DINOv3 weights/LoRA injection are hidden
    in this core; use the preceding adapter exercise to supply the encoder.
    """

    class Restoration(nn.Module):

        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.sr = nn.Sequential(
                nn.Conv2d(channels, 3 * sr_scale**2, 3, padding=1),
                nn.PixelShuffle(sr_scale),
            )
            self.enh = nn.Conv2d(channels, 3, 3, padding=1)

        def forward(self, x, task="sr"):
            b, _, h, w = x.shape
            if h % patch_size or w % patch_size:
                raise ValueError("patch divisibility")
            tokens = self.encoder(x)
            if tokens.shape != (
                b,
                h // patch_size * (w // patch_size),
                channels,
            ):
                raise ValueError("patch token shape")
            grid = tokens.transpose(1, 2).reshape(
                b, channels, h // patch_size, w // patch_size
            )
            features = F.interpolate(
                grid, size=(h, w), mode="bilinear", align_corners=False
            )
            if task == "sr":
                return q514(x, self.sr(features), "sr")
            if task == "enhancement":
                return q514(x, self.enh(features), "enhancement")
            raise ValueError("unknown task")

    return Restoration()
