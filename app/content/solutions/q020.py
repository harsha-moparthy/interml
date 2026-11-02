def q020(series, level=-1):
    """Move an index level into columns and fill missing combinations with zero."""
    return series.unstack(level=level, fill_value=0)
