import pandas as pd


def q548(observed, eligibility):
    """Build a small eligible entity-quarter panel. observed has entity,quarter,value;
    eligibility has one entity row with inclusive integer start/end quarters. Return all
    eligible rows with observed flag; preserve observed zero versus missing NaN. Reject
    duplicate keys/entities. Rows outside eligibility are excluded.
    """
    if (
        observed.duplicated(["entity", "quarter"]).any()
        or eligibility["entity"].duplicated().any()
    ):
        raise ValueError("duplicate keys")
    rows = []
    for row in eligibility.itertuples(index=False):
        if row.end < row.start:
            raise ValueError("invalid eligibility interval")
        rows.extend(((row.entity, q) for q in range(row.start, row.end + 1)))
    panel = pd.DataFrame(rows, columns=["entity", "quarter"])
    result = panel.merge(
        observed,
        on=["entity", "quarter"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    result["observed"] = result.pop("_merge").eq("both")
    return result
