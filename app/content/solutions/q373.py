def q373(candidate, reference):
    """Implement an exact-token METEOR variant for token lists. Greedily align each
    candidate token to the first unused identical reference token. Use 10PR/(R+9P) and
    penalty 0.5*(chunks/matches)^3; omit stemming and synonyms.
    """
    used = set()
    alignment = []
    for i, token in enumerate(candidate):
        for j, other in enumerate(reference):
            if j not in used and token == other:
                used.add(j)
                alignment.append((i, j))
                break
    m = len(alignment)
    if not m:
        return 0.0
    p = m / len(candidate)
    r = m / len(reference)
    chunks = 1 + sum(
        (i != a + 1 or j != b + 1 for (a, b), (i, j) in zip(alignment, alignment[1:]))
    )
    return 10 * p * r / (r + 9 * p) * (1 - 0.5 * (chunks / m) ** 3)
