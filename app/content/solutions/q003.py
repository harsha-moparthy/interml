def q003(frame, row_label, row_position):
    """Return independent rows selected by label and by zero-based position."""
    return (frame.loc[row_label].copy(), frame.iloc[row_position].copy())
