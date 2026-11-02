def q399(tree, next_argmax, prompt, max_tokens):
    """Verify a speculative token tree by greedy target predictions. tree maps token IDs to
    child dictionaries. Emit matched edges and the first correction outside the tree,
    bounded by max_tokens.
    """
    history = list(prompt)
    emitted = []
    node = tree
    for _ in range(max_tokens):
        token = int(next_argmax(history))
        emitted.append(token)
        history.append(token)
        if token not in node:
            break
        node = node[token]
    return emitted
