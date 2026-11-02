def q397(draft_tokens, target_argmax):
    """Verify a greedy draft against target argmax tokens for each draft position plus one
    bonus position. Return accepted prefix plus the first correction, or all drafts plus
    the bonus.
    """
    if len(target_argmax) != len(draft_tokens) + 1:
        raise ValueError("one bonus prediction required")
    result = []
    for draft, target in zip(draft_tokens, target_argmax):
        result.append(int(target))
        if draft != target:
            return result
    result.append(int(target_argmax[-1]))
    return result
