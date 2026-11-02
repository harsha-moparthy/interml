def q383(text, required=(), forbidden=(), max_words=None, exact_lines=None):
    """Check literal case-sensitive required/forbidden substrings, a whitespace-token word
    limit, and an exact splitlines count. Return individual Boolean results and their
    conjunction.
    """
    checks = {
        "required": all((s in text for s in required)),
        "forbidden": all((s not in text for s in forbidden)),
        "max_words": max_words is None or len(text.split()) <= max_words,
        "exact_lines": exact_lines is None or len(text.splitlines()) == exact_lines,
    }
    return (all(checks.values()), checks)
