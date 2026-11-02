from torch.nn import functional as F


def q268(x, kernel=2, stride=2):
    """Perform nonpadded 2D max pooling on NCHW tensors using unfold. Ignore incomplete
    boundary windows.
    """
    n, c, h, w = x.shape
    patches = F.unfold(x, kernel, stride=stride).reshape(n, c, kernel * kernel, -1)
    return patches.max(dim=2).values.reshape(
        n, c, (h - kernel) // stride + 1, (w - kernel) // stride + 1
    )
