import math
import numpy as np


def q546(actual, predicted, groups=None):
    y = np.asarray(actual, float)
    p = np.asarray(predicted, float)
    if y.shape != p.shape or y.ndim != 1:
        raise ValueError("expected matching vectors")
    g = np.zeros(len(y), int) if groups is None else np.asarray(groups)
    if len(g) != len(y):
        raise ValueError("group length")
    valid = np.isfinite(y) & np.isfinite(p)

    def score(mask):
        denominator = np.abs(y[mask]).sum()
        return (
            np.abs(y[mask] - p[mask]).sum() / denominator
            if denominator
            else float("nan")
        )

    values = {key: score(valid & (g == key)) for key in dict.fromkeys(g.tolist())}
    defined = [v for v in values.values() if np.isfinite(v)]
    return (
        score(valid),
        values,
        float(np.mean(defined)) if defined else float("nan"),
    )


def q553(series, cutoffs, horizon, forecast):
    values = np.asarray(series, float)
    result = []
    if horizon < 1:
        raise ValueError("positive horizon required")
    for cutoff in cutoffs:
        if cutoff < 1 or cutoff + horizon > len(values):
            continue
        prediction = np.asarray(forecast(values[:cutoff].copy(), horizon), float)
        if prediction.shape != (horizon,):
            raise ValueError("forecast horizon mismatch")
        truth = values[cutoff : cutoff + horizon].copy()
        result.append((cutoff, prediction, truth, q546(truth, prediction)[0]))
    return result


def q554(partitions):
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


def q555(series, cutoffs, horizon, season=4, window=4):
    """Benchmark seasonal-naive and trailing-mean forecasts under the SAME rolling origins.
    Skip origins lacking max(season,window) history; return records and pooled
    forecast-cell WAPE for each model. Overlapping horizons count each forecast cell
    separately. Data deduplication/as-of joins are preceding exercises, not hidden in
    this benchmark core.
    """
    if season < 1 or window < 1:
        raise ValueError("positive season/window required")
    cutoffs = [c for c in cutoffs if c >= max(season, window)]
    models = {
        "seasonal": lambda x, h: np.resize(x[-season:], h),
        "mean": lambda x, h: np.full(h, x[-window:].mean()),
    }
    results = {}
    for name, model in models.items():
        records = q553(series, cutoffs, horizon, model)
        _, metrics = q554([(row[2], row[1]) for row in records])
        results[name] = {"records": records, "wape": metrics["wape"]}
    return results
