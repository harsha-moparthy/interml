def q006(frame, fill_values):
    """Fill missing entries using a scalar or per-column mapping; preserve the input."""
    return frame.fillna(fill_values)
