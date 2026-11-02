from torch.nn import functional as F


def q580(image, mask, size, flip=False, minimum_area=0.1):
    """Resize CHW image and HW categorical mask together, bilinear for image and nearest
    for mask; optionally flip both horizontally. Return image, integer mask,
    nonzero-mask area fraction, and inclusion flag. This deterministic transform core
    treats all nonzero labels as organ and preserves class IDs.
    """
    if image.shape[-2:] != mask.shape or not 0 <= minimum_area <= 1:
        raise ValueError("invalid pair")
    result = F.interpolate(
        image[None], size=size, mode="bilinear", align_corners=False
    )[0]
    labels = F.interpolate(mask[None, None].float(), size=size, mode="nearest")[
        0, 0
    ].to(mask.dtype)
    if flip:
        result = result.flip(-1)
        labels = labels.flip(-1)
    area = float((labels != 0).float().mean())
    return (result, labels, area, area >= minimum_area)
