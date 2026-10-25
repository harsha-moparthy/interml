"""Remove all authored animation/demo payloads from lesson JSON sources."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    changed = 0
    for path in sorted((ROOT / "app/content/articles").glob("*.json")):
        article = json.loads(path.read_text())
        blocks = article.get("blocks", [])
        clean = [b for b in blocks if b.get("type") not in {"demonstration", "compare"}]
        if clean == blocks and not article.get("demonstrations"):
            continue
        article["blocks"] = clean
        article["demonstrations"] = []
        article["revision"] = int(article.get("revision", 0)) + 1
        path.write_text(json.dumps(article, indent=2, ensure_ascii=False) + "\n")
        changed += 1
    print(f"removed animation data from {changed} articles")
    return changed


if __name__ == "__main__":
    main()
