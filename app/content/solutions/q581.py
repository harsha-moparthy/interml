import torch
from torch import nn
from torch.nn import functional as F


def q581(channels=1, width=8):
    """Return a small two-level U-Net producing one logit per input pixel. Use bilinear
    interpolation to align each upsampled tensor with its skip, supporting odd H/W>=4.
    Encoder widths are w,2w,4w; this complete small network extends an isolated
    skip-connection exercise.
    """

    def block(a, b):
        return nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(b, b, 3, padding=1),
            nn.ReLU(),
        )

    class UNet(nn.Module):

        def __init__(self):
            super().__init__()
            self.e1 = block(channels, width)
            self.e2 = block(width, 2 * width)
            self.mid = block(2 * width, 4 * width)
            self.d2 = block(6 * width, 2 * width)
            self.d1 = block(3 * width, width)
            self.out = nn.Conv2d(width, 1, 1)

        def forward(self, x):
            if min(x.shape[-2:]) < 4:
                raise ValueError("image too small")
            a = self.e1(x)
            b = self.e2(F.max_pool2d(a, 2))
            z = self.mid(F.max_pool2d(b, 2))
            z = self.d2(
                torch.cat(
                    (
                        F.interpolate(
                            z,
                            size=b.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ),
                        b,
                    ),
                    1,
                )
            )
            z = self.d1(
                torch.cat(
                    (
                        F.interpolate(
                            z,
                            size=a.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        ),
                        a,
                    ),
                    1,
                )
            )
            return self.out(z)

    return UNet()
