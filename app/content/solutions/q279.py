import torch
from torch.nn import functional as F


def q279(feature, box, output_size=3):
    """Implement a simple ROI Align variant for a single CHW feature map: sample one point
    at each output bin center inside a continuous xyxy box. Use bilinear interpolation
    with border clamping, no half-pixel shift.
    """
    c, h, w = feature.shape
    x1, y1, x2, y2 = box
    ys = (
        y1
        + (torch.arange(output_size, device=feature.device, dtype=feature.dtype) + 0.5)
        * (y2 - y1)
        / output_size
    )
    xs = (
        x1
        + (torch.arange(output_size, device=feature.device, dtype=feature.dtype) + 0.5)
        * (x2 - x1)
        / output_size
    )
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    gx = 2 * gx / max(w - 1, 1) - 1
    gy = 2 * gy / max(h - 1, 1) - 1
    grid = torch.stack([gx, gy], dim=-1)[None]
    return F.grid_sample(
        feature[None], grid, align_corners=True, padding_mode="border"
    )[0]
