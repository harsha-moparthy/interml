import numpy as np


def q127(rows, buckets):
    """Hash sparse feature dictionaries into signed dense vectors using stable BLAKE2
    hashes. Collisions add, and the result is independent of Python's randomized hash
    seed.
    """
    import hashlib

    if buckets <= 0:
        raise ValueError("positive bucket count required")
    result = np.zeros((len(rows), buckets))
    for i, row in enumerate(rows):
        for name, value in row.items():
            digest = hashlib.blake2b(str(name).encode(), digest_size=16).digest()
            slot = int.from_bytes(digest[:8], "little") % buckets
            sign = 1 if digest[8] & 1 else -1
            result[i, slot] += sign * value
    return result
