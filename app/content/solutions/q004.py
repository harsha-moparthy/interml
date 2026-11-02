def q004(frame, column, low, high):
    """Keep rows inside the inclusive interval [low, high]; exclude missing values."""
    mask = frame[column].between(low, high).fillna(False)
    return frame.loc[mask].copy()
