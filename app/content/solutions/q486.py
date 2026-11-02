def q486(tokens, size, overlap=0):
    """Split a token sequence into overlapping chunks of at most size tokens. Require
    0<=overlap<size; emit no empty chunks.
    """
    if not 0 <= overlap < size:
        raise ValueError("invalid overlap")
    result = []
    for start in range(0, len(tokens), size - overlap):
        result.append(tokens[start : start + size])
        if start + size >= len(tokens):
            break
    return result
