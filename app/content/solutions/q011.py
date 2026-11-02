def q011(frame, group, value):
    """Sum values per group in encounter order, including a group for missing keys."""
    return frame.groupby(group, dropna=False, sort=False)[value].sum().reset_index()
