import numpy as np


def q552(targets, forecasts, quantiles):
    """Evaluate [N,H,Q] quantile forecasts against [N,H] targets. On rows with finite
    target and all forecast quantiles, return mean pinball loss, outer-quantile interval
    coverage/width, and crossed-row count. Quantiles must strictly increase inside
    (0,1); undefined empty metrics are NaN.
    """
    y = np.asarray(targets, float)
    f = np.asarray(forecasts, float)
    q = np.asarray(quantiles, float)
    if (
        f.shape != (*y.shape, len(q))
        or len(q) < 2
        or np.any(np.diff(q) <= 0)
        or np.any((q <= 0) | (q >= 1))
    ):
        raise ValueError("invalid quantiles/shapes")
    valid = np.isfinite(y) & np.isfinite(f).all(-1)
    yy = y[valid]
    ff = f[valid]
    if not len(yy):
        return {
            "pinball": np.nan,
            "coverage": np.nan,
            "width": np.nan,
            "crossed": 0,
        }
    error = yy[:, None] - ff
    return {
        "pinball": float(np.maximum(q * error, (q - 1) * error).mean()),
        "coverage": float(((yy >= ff[:, 0]) & (yy <= ff[:, -1])).mean()),
        "width": float((ff[:, -1] - ff[:, 0]).mean()),
        "crossed": int((np.diff(ff, axis=1) < 0).any(1).sum()),
    }
