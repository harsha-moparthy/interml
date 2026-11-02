import numpy as np


def q411(samples, percentile=99.9):
    """Calibrate symmetric INT8 scales from absolute calibration samples using AbsMax and a
    chosen percentile. Return scales; zero thresholds use scale 1.
    """
    a = np.abs(np.asarray(samples, float))
    thresholds = [a.max(), np.percentile(a, percentile)]
    return {
        name: float(t / 127 if t else 1)
        for name, t in zip(("absmax", "percentile"), thresholds)
    }
