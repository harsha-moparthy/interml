import heapq


def q585(
    initial,
    expand,
    score,
    verify,
    max_depth=3,
    branch=2,
    max_verifications=10,
):
    """Budgeted best-first repair search over hashable states. expand(state)->iterable;
    score is larger-is-better; verify returns Boolean. Limit depth, children per
    expansion, and verifier calls; deduplicate states on insertion. Return first
    verified state or None plus verification trace. This core models call/depth budgets,
    not tokens or a real program-analysis proof.
    """
    if min(max_depth, branch, max_verifications) < 0:
        raise ValueError("negative budget")
    queue = [(-score(initial), 0, 0, initial)]
    seen = {initial}
    serial = 1
    trace = []
    while queue and len(trace) < max_verifications:
        _, _, depth, state = heapq.heappop(queue)
        passed = bool(verify(state))
        trace.append((state, passed))
        if passed:
            return (state, trace)
        if depth == max_depth:
            continue
        children = iter(expand(state))
        for _ in range(branch):
            try:
                child = next(children)
            except StopIteration:
                break
            if child in seen:
                continue
            seen.add(child)
            heapq.heappush(queue, (-score(child), serial, depth + 1, child))
            serial += 1
    return (None, trace)
