def q395(cached_prefixes, query):
    """Build a token trie from (token_sequence,cache_handle) entries and return the longest
    complete cached prefix of query as (matched_length,handle). A handle may be any
    object.
    """
    root = {}
    marker = object()
    for tokens, handle in cached_prefixes:
        node = root
        for token in tokens:
            node = node.setdefault(token, {})
        node[marker] = handle
    best = (0, root.get(marker))
    node = root
    for i, token in enumerate(query):
        if token not in node:
            break
        node = node[token]
        if marker in node:
            best = (i + 1, node[marker])
    return best
