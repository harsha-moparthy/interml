"""Reference values for the broadcasting demonstration, computed with NumPy.
Usage: python tools/gen_broadcast_expected.py > app/frontend/tests/broadcast_expected.json
A holds 1, 2, 3, ... and B holds 10, 20, 30, ... in row-major order (same rule as broadcast.js)."""
import itertools, json, random
import numpy as np

def case(sa, sb):
    A = np.arange(1, 1 + int(np.prod(sa))).reshape(sa)
    B = 10 * np.arange(1, 1 + int(np.prod(sb))).reshape(sb)
    try:
        C = A + B
    except ValueError:
        pa = [1] * (3 - len(sa)) + list(sa); pb = [1] * (3 - len(sb)) + list(sb)
        conflict = [3 - d for d in range(3) if pa[d] != pb[d] and 1 not in (pa[d], pb[d])]
        return {"inputs": {"shapeA": list(sa), "shapeB": list(sb)},
                "expect": {"valid": False, "result_shape": None, "conflict_from_right": conflict, "output": None}}
    out = C.reshape([1] * (3 - C.ndim) + list(C.shape))
    return {"inputs": {"shapeA": list(sa), "shapeB": list(sb)},
            "expect": {"valid": True, "result_shape": list(out.shape), "conflict_from_right": [], "output": out.tolist()}}

if __name__ == "__main__":
    rng = random.Random(0)
    shapes = [s for n in (1, 2, 3) for s in itertools.product(range(1, 5), repeat=n)]
    pairs = [((2, 1, 3), (1, 4, 1)), ((3,), (4,)), ((3,), (3, 1)), ((2, 1, 3), (2, 3))]
    pairs += [(rng.choice(shapes), rng.choice(shapes)) for _ in range(300)]
    print(json.dumps([case(a, b) for a, b in pairs]))
