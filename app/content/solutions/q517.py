from torch.nn import functional as F


def q517(x, y, window=3, data_range=1.0):
    """Return mean local SSIM for equal BCHW images using a uniform VALID window,
    population moments, C1=(.01*range)^2 and C2=(.03*range)^2. There is no border
    padding. This uniform-window variant must not be confused with a Gaussian-window
    reference.
    """
    if (
        x.shape != y.shape
        or window < 1
        or window > min(x.shape[-2:])
        or (data_range <= 0)
    ):
        raise ValueError("invalid shape/window/range")
    pool = lambda z: F.avg_pool2d(z, window, stride=1)
    mx, my = (pool(x), pool(y))
    vx = (pool(x * x) - mx * mx).clamp_min(0)
    vy = (pool(y * y) - my * my).clamp_min(0)
    cov = pool(x * y) - mx * my
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    return (
        (2 * mx * my + c1)
        * (2 * cov + c2)
        / ((mx * mx + my * my + c1) * (vx + vy + c2))
    ).mean()
