def q005(frame, keys):
    """Create an index from keys, rejecting duplicate index entries."""
    return frame.set_index(keys, verify_integrity=True)
