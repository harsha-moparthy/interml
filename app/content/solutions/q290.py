from collections import Counter


def q290(tokens, min_count=1):
    """Build token-to-ID and ID-to-token vocabularies, sorting eligible tokens by
    descending frequency then lexicographically.
    """
    counts = Counter(tokens)
    ordered = sorted(
        (t for t, c in counts.items() if c >= min_count),
        key=lambda t: (-counts[t], t),
    )
    return ({token: i for i, token in enumerate(ordered)}, ordered)
