from collections import OrderedDict


def q396(requests, budget_bytes):
    """Simulate LRU caching from (key,size_bytes) accesses under a byte budget. Oversized
    objects bypass caching. Return hits, evicted keys, and final oldest-to-newest keys;
    sizes for an existing key must be unchanged.
    """
    cache = OrderedDict()
    used = 0
    hits = 0
    evicted = []
    for key, size in requests:
        if size < 0:
            raise ValueError("negative size")
        if key in cache:
            hits += 1
            cache.move_to_end(key)
            continue
        if size > budget_bytes:
            continue
        while used + size > budget_bytes:
            old, old_size = cache.popitem(last=False)
            used -= old_size
            evicted.append(old)
        cache[key] = size
        used += size
    return (hits, evicted, list(cache))
