def q549(requests, versions):
    """As-of feature-join reference. Requests are (entity,cutoff); versions are dicts with
    entity,event_time,available_time,version,value. Select the greatest
    (event_time,available_time,version) eligible tuple where BOTH timestamps <=cutoff.
    Return selected row copies or None in request order. This O(R*V) core prioritizes
    the temporal contract.
    """
    output = []
    for entity, cutoff in requests:
        eligible = [
            v
            for v in versions
            if v["entity"] == entity
            and v["event_time"] <= cutoff
            and (v["available_time"] <= cutoff)
        ]
        row = (
            max(
                eligible,
                key=lambda v: (
                    v["event_time"],
                    v["available_time"],
                    v["version"],
                ),
            )
            if eligible
            else None
        )
        output.append(dict(row) if row is not None else None)
    return output
