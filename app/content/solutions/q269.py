from torch.nn import functional as F


def q269(x, kernel=2, stride=2):
    """Perform nonpadded 2D average pooling on NCHW tensors using unfold. Average only
    complete windows.
    """
    n, c, h, w = x.shape
    patches = F.unfold(x, kernel, stride=stride).reshape(n, c, kernel * kernel, -1)
    return patches.mean(dim=2).reshape(
        n, c, (h - kernel) // stride + 1, (w - kernel) // stride + 1
    )
