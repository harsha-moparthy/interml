import numpy as np


def q550(training_prices, future_prices, bands=4):
    """Fit quantile band edges on finite historical prices and apply unchanged to future
    prices. Interior boundaries go to the HIGHER band, duplicate edges collapse, finite
    out-of-range values enter extreme bands, and missing/nonfinite prices map to -1.
    Return edges and integer band IDs.
    """
    historical = np.asarray(training_prices, float)
    historical = historical[np.isfinite(historical)]
    if not len(historical) or bands < 1:
        raise ValueError("invalid history/bands")
    edges = np.unique(np.quantile(historical, np.linspace(0, 1, bands + 1)[1:-1]))
    values = np.asarray(future_prices, float)
    assigned = np.searchsorted(edges, values, side="right")
    return (edges, np.where(np.isfinite(values), assigned, -1))
