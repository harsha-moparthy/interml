def q019(frame, id_columns, value_columns):
    """Convert selected value columns to variable/value rows, keeping identifier columns."""
    return frame.melt(
        id_vars=id_columns,
        value_vars=value_columns,
        var_name="variable",
        value_name="value",
    )
