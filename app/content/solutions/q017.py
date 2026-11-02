def q017(left, right, keys, how="left"):
    """Join on keys, rejecting duplicate keys on the right to prevent row expansion."""
    return left.merge(right, on=keys, how=how, sort=False, validate="many_to_one")
