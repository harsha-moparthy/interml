def q010(frame, dtypes):
    """Convert columns to the requested dtypes; invalid conversions raise errors."""
    return frame.astype(dtypes)
