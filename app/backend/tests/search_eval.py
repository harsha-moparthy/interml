"""Compare keyword-only, semantic-only, and hybrid search on tests/search_eval.json.
Run: python tests/search_eval.py   Prints hit@5 (a relevant question in the top 5) and mean reciprocal rank."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from placement import search  # noqa: E402

import os
CASES = json.loads((Path(__file__).parent / os.environ.get("EVAL_SET", "search_eval.json")).read_text())


def evaluate(mode, floor=None):
    if floor is not None:
        search.SEMANTIC_FLOOR = floor
    hits = rr = 0
    misses = []
    for c in CASES:
        ids = search.search(c["q"], mode=mode)["ids"]
        pos = next((n for n, i in enumerate(ids) if i in c["rel"]), None)
        if pos is not None and pos < 5:
            hits += 1
        else:
            misses.append((c["q"], ids[:3]))
        rr += 1 / (pos + 1) if pos is not None else 0
    return hits / len(CASES), rr / len(CASES), misses


if __name__ == "__main__":
    for mode in ("keyword", "semantic", "hybrid"):
        h, m, misses = evaluate(mode)
        print(f"{mode:9s} hit@5={h:.2f}  MRR={m:.2f}  misses={len(misses)}")
        if "-v" in sys.argv:
            for q, got in misses:
                print(f"    miss: {q!r} -> {got}")
