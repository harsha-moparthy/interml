def q012(frame, column):
    """Summarize a column; count excludes missing values and std uses ddof=1."""
    return frame[column].agg(["count", "mean", "min", "max", "std"])
