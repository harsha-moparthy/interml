import numpy as np


def q547(events):
    """Deduplicate a DataFrame by event_id keeping greatest version (ties: last input row).
    Group by node,band,quarter including null keys; sum sessions/purchases and recompute
    conversion. Return grouped table and input/deduplicated/output row counts. This
    small Pandas reference models a distributed aggregation contract.
    """
    clean = events.sort_values("version", kind="stable").drop_duplicates(
        "event_id", keep="last"
    )
    result = (
        clean.groupby(["node", "band", "quarter"], dropna=False)[
            ["sessions", "purchases"]
        ]
        .sum()
        .reset_index()
    )
    result["conversion"] = result["purchases"] / result["sessions"].replace(0, np.nan)
    return (
        result,
        {
            "input": len(events),
            "deduplicated": len(clean),
            "groups": len(result),
        },
    )
