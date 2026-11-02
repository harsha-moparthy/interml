def q008(frame, mapping):
    """Rename columns; raise KeyError when a requested source column is absent."""
    return frame.rename(columns=mapping, errors="raise")
