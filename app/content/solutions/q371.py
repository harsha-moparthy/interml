import math
from collections import Counter


def q371(candidate, references, max_order=4):
    """Compute unsmoothed sentence BLEU for token lists and one or more references. Use
    clipped counts, uniform weights through max_order, and the closest reference length
    (shorter breaks ties). Missing n-grams give zero.
    """
    if not references or max_order < 1:
        raise ValueError("references and positive order required")
    c = len(candidate)
    if c == 0:
        return 0.0
    precisions = []
    for n in range(1, max_order + 1):
        counts = Counter((tuple(candidate[i : i + n]) for i in range(c - n + 1)))
        maximum = Counter()
        for ref in references:
            maximum |= Counter((tuple(ref[i : i + n]) for i in range(len(ref) - n + 1)))
        matches = sum((min(count, maximum[gram]) for gram, count in counts.items()))
        total = sum(counts.values())
        if not matches or not total:
            return 0.0
        precisions.append(matches / total)
    r = min(
        (len(ref) for ref in references),
        key=lambda length: (abs(length - c), length),
    )
    return math.exp(min(0, 1 - r / c) + sum(map(math.log, precisions)) / max_order)
