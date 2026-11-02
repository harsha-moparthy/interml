def q340(embedding, head):
    """Tie an nn.Linear output head's weight to an nn.Embedding's weight so both reference
    the same Parameter. Require identical shapes and return the head.
    """
    if head.weight.shape != embedding.weight.shape:
        raise ValueError("shape mismatch")
    head.weight = embedding.weight
    return head
