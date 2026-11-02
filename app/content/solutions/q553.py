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
    """Rolling-origin evaluator for a single numeric series. Cutoff is an exclusive context
    endpoint; supplied forecast(context_copy,horizon) returns a horizon vector. Skip
    origins lacking history or full future targets; return
    (cutoff,prediction,actual,WAPE) records. Only past context is passed; no covariates
    or model downloads.
    """
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
