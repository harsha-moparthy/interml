def q330(token_embeddings, position_table, offset=0):
    """Add learned positional embeddings to a (B,L,D) token embedding tensor, starting at a
    supplied offset. Reject positions outside the table.
    """
    length = token_embeddings.shape[1]
    if offset < 0 or offset + length > len(position_table):
        raise ValueError("position out of range")
    return token_embeddings + position_table[offset : offset + length][None, :, :]
