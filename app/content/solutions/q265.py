import torch
from torch.nn import functional as F


def q265(x, weights, bias=None, dilation=2, padding=0, stride=1):
    """Implement multi-channel dilated 2D convolution by unfold and contraction. X is NCHW
    and weights is (out_channels,in_channels,kh,kw); PyTorch's high-level conv2d is not
    used.
    """
    n, c, h, w = x.shape
    out, _, kh, kw = weights.shape
    columns = F.unfold(x, (kh, kw), dilation=dilation, padding=padding, stride=stride)
    values = torch.einsum("oc,ncl->nol", weights.reshape(out, -1), columns)
    if bias is not None:
        values = values + bias[None, :, None]
    oh = (h + 2 * padding - dilation * (kh - 1) - 1) // stride + 1
    ow = (w + 2 * padding - dilation * (kw - 1) - 1) // stride + 1
    return values.reshape(n, out, oh, ow)
