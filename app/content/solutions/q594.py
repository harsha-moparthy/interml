import copy


def q594(left, right, candidates, k=5):
    """Merge two sketches created by the preceding Count-Min implementation into a copy,
    requiring matching width/depth/hash seed. Return merged sketch and estimated top-k
    over a supplied candidate-key set. This candidate-based TopK is approximate and does
    not enumerate keys from counters.
    """
    if (left.width, left.depth, left.seed) != (
        right.width,
        right.depth,
        right.seed,
    ) or k < 0:
        raise ValueError("incompatible sketches/k")
    merged = copy.deepcopy(left)
    merged.table = [
        [a + b for a, b in zip(left_row, right_row)]
        for left_row, right_row in zip(left.table, right.table)
    ]
    values = [(key, merged.estimate(key)) for key in set(candidates)]
    return (merged, sorted(values, key=lambda p: (-p[1], p[0]))[:k])
