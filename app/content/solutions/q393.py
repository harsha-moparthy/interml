def q393(token_counts, block_size, total_blocks):
    """Allocate disjoint physical block IDs to requests with specified token counts. Return
    block tables and remaining IDs; fail atomically when capacity is insufficient.
    """
    if block_size < 1 or any((n < 0 for n in token_counts)):
        raise ValueError("invalid sizes")
    counts = [(n + block_size - 1) // block_size for n in token_counts]
    if sum(counts) > total_blocks:
        raise MemoryError("insufficient blocks")
    tables = []
    cursor = 0
    for count in counts:
        tables.append(list(range(cursor, cursor + count)))
        cursor += count
    return (tables, list(range(cursor, total_blocks)))
