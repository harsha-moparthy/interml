import math
import numpy as np


def q554(partitions):
    """Merge sufficient statistics from iterable (actual,prediction) partitions. Return
    [sum_abs_error,sum_abs_target,sum_squared_error,count] and WAPE/MAE/RMSE; finite
    paired observations only. Empty denominators yield NaN. Never average partition RMSE
    or WAPE directly.
    """
    total = np.zeros(4, dtype=np.float64)
    for actual, predicted in partitions:
        y = np.asarray(actual, float)
        p = np.asarray(predicted, float)
        if y.shape != p.shape:
            raise ValueError("shape mismatch")
        mask = np.isfinite(y) & np.isfinite(p)
        error = y[mask] - p[mask]
        total += np.array(
            [
                np.abs(error).sum(),
                np.abs(y[mask]).sum(),
                np.square(error).sum(),
                mask.sum(),
            ]
        )
    ae, ay, se, n = total
    return (
        total,
        {
            "wape": ae / ay if ay else np.nan,
            "mae": ae / n if n else np.nan,
            "rmse": math.sqrt(se / n) if n else np.nan,
        },
    )
