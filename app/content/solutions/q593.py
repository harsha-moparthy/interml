import hashlib


def q593(width=64, depth=4, seed=0):
    """Return a sequential Count-Min Sketch with deterministic row-seeded BLAKE2b hashes,
    nonnegative integer updates, and minimum-counter estimates. Python integers avoid
    counter overflow. Byte keys are hashed directly, string keys as UTF-8. It is a
    reference model, not the resume's lock-free C++ implementation.
    """
    if min(width, depth) < 1:
        raise ValueError("positive dimensions")

    class Sketch:

        def __init__(self):
            self.width = width
            self.depth = depth
            self.seed = seed
            self.table = [[0] * width for _ in range(depth)]

        def positions(self, key):
            payload = key if isinstance(key, bytes) else key.encode("utf-8")
            return [
                int.from_bytes(
                    hashlib.blake2b(
                        payload,
                        person=hashlib.sha256(f"{seed}:{row}".encode()).digest()[:16],
                        digest_size=8,
                    ).digest(),
                    "little",
                )
                % width
                for row in range(depth)
            ]

        def update(self, key, count=1):
            if not isinstance(count, int) or count < 0:
                raise ValueError("nonnegative integer updates only")
            for row, column in enumerate(self.positions(key)):
                self.table[row][column] += count

        def estimate(self, key):
            return min(
                (
                    self.table[row][column]
                    for row, column in enumerate(self.positions(key))
                )
            )

    return Sketch()
