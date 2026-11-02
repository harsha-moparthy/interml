def q015(frame, keys, value):
    """Return count, sum, and mean for each key tuple, including missing keys."""
    return (
        frame.groupby(keys, dropna=False)[value]
        .agg(["count", "sum", "mean"])
        .reset_index()
    )
