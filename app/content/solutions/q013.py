def q013(frame, rows, columns, values):
    """Pivot to a mean-valued table, filling missing result cells with zero."""
    return frame.pivot_table(
        index=rows,
        columns=columns,
        values=values,
        aggfunc="mean",
        fill_value=0,
    )
