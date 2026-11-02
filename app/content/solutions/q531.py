def q531(candidates, positives, count, self_id=None):
    """Select up to count unique hard-negative IDs in supplied rank order. Exclude all
    known positives and optional self ID. Return fewer if insufficient; no fallback
    samples or unknown relevance labels are fabricated.
    """
    if count < 0:
        raise ValueError("negative count")
    blocked = set(positives)
    selected = []
    if self_id is not None:
        blocked.add(self_id)
    for item in candidates:
        if len(selected) == count:
            break
        if item not in blocked:
            selected.append(item)
            blocked.add(item)
    return selected
