import numpy as np


def q306(shingles, hashes=64):
    """Build deterministic MinHash signatures from sets of string shingles using seeded
    BLAKE2 digests. Empty sets receive the maximum uint64 value in every coordinate.
    """
    import hashlib

    signature = np.full(hashes, np.iinfo(np.uint64).max, dtype=np.uint64)
    for token in set(shingles):
        for i in range(hashes):
            digest = hashlib.blake2b(
                token.encode(), digest_size=8, key=i.to_bytes(8, "little")
            ).digest()
            signature[i] = min(signature[i], int.from_bytes(digest, "little"))
    return signature
