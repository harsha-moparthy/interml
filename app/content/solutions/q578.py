import numpy as np


def q578(raw, slope=1, intercept=0, lower=-100, upper=400, gamma=1):
    """Convert raw CT pixels to HU using slope/intercept, clip to [lower,upper], normalize
    to [0,1], and apply x^gamma. Return a floating array; masks are not inputs and
    cannot be modified. Require upper>lower and gamma>0.
    """
    if upper <= lower or gamma <= 0:
        raise ValueError("invalid window/gamma")
    hu = np.asarray(raw, float) * slope + intercept
    return ((np.clip(hu, lower, upper) - lower) / (upper - lower)) ** gamma
