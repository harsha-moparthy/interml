from torch.nn import functional as F


def q266(x, depthwise, pointwise, padding=0):
    """Apply depthwise convolution with one kernel per input channel, then a 1x1 pointwise
    channel projection. Use PyTorch convolution primitives.
    """
    spatial = F.conv2d(x, depthwise, padding=padding, groups=x.shape[1])
    return F.conv2d(spatial, pointwise)
