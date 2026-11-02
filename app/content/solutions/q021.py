def q021(frame, column, transform):
    """Map nonmissing column values through transform on an independent frame."""
    result = frame.copy()
    result[column] = result[column].map(transform, na_action="ignore")
    return result
