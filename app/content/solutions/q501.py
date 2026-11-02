import torch
from torch.nn import functional as F


def q501(images, sizes=(32, 32, 16), scales=(0.8, 0.8, 0.4), seed=0):
    """Generate independent square crops for each BCHW image/view, resize bilinearly, and
    return view batches and (top,left,side) metadata. Sizes/scales are equal-length
    positive sequences; the first two views are global. Scales specify crop side as a
    fraction of min(H,W). This core omits photometric augmentation.
    """
    if len(sizes) < 2 or len(sizes) != len(scales):
        raise ValueError("need two globals and matching sizes/scales")
    b, c, h, w = images.shape
    rng = torch.Generator().manual_seed(seed)
    views, metadata = ([], [])
    for size, scale in zip(sizes, scales):
        if size < 1 or not 0 < scale <= 1:
            raise ValueError("invalid size/scale")
        side = max(1, int(min(h, w) * scale))
        crops, coordinates = ([], [])
        for image in images:
            top = int(torch.randint(h - side + 1, (), generator=rng))
            left = int(torch.randint(w - side + 1, (), generator=rng))
            crop = image[:, top : top + side, left : left + side]
            crops.append(
                F.interpolate(
                    crop[None],
                    (size, size),
                    mode="bilinear",
                    align_corners=False,
                )[0]
            )
            coordinates.append((top, left, side))
        views.append(torch.stack(crops))
        metadata.append(coordinates)
    return (views, metadata)
