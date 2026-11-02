import hashlib


def q584(candidates, verify):
    """Verify exact-text repair candidates at most once per SHA256 digest. Preserve every
    input occurrence with digest/result metadata. UTF-8 source text is hashed unchanged;
    no whitespace/comment canonicalization that could change semantics. verify is a
    supplied pure/mock callback.
    """
    cache = {}
    results = []
    for source in candidates:
        key = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if key not in cache:
            cache[key] = verify(source)
        results.append({"source": source, "hash": key, "result": cache[key]})
    return results
