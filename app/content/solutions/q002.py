def q002(frame, columns):
    """Select columns in the requested order and return an independent frame."""
    return frame.loc[:, columns].copy()
