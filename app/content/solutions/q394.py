def q394(blocks, refcounts, table, logical_block, free_ids):
    """Ensure a request's logical block is privately writable. Mutate block storage,
    reference counts, table, and free-ID list only if a shared block needs copying;
    return the physical ID.
    """
    old = table[logical_block]
    if refcounts[old] > 1:
        if not free_ids:
            raise MemoryError("no free block")
        new = free_ids.pop()
        blocks[new] = blocks[old].copy()
        refcounts[new] = 1
        refcounts[old] -= 1
        table[logical_block] = new
    return table[logical_block]
