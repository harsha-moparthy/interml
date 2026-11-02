def q007(frame, keys):
    """Keep the first row for each key combination, preserving encounter order."""
    return frame.drop_duplicates(subset=keys, keep="first").copy()
