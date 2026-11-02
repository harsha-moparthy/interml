def q016(frame, group, value):
    """Attach each row’s group mean without changing row order or the input."""
    result = frame.copy()
    # transform broadcasts one aggregate back to every original row.
    result["group_mean"] = frame.groupby(group, dropna=False)[value].transform("mean")
    return result
