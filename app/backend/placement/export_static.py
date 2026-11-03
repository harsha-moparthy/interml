"""Write every read-only API response as a JSON file, so the site can be served as plain static files (GitHub Pages).

    cd app/backend && ../.venv/bin/python -m placement.export_static

Output goes to app/frontend/public/static-api/. The folder mirrors the API routes:
chapters.json, stats.json, all.json, articles.json, articles/<id>.json, exercises/<id>.json, exercises/<id>/solution.json,
search-index.json for the in-browser search, and embeddings.bin (the stored meaning vectors). Run it again after any import.
"""
import json
import shutil
import sys

from fastapi.testclient import TestClient

from . import config, db, search
from .main import app

OUT = config.APP_DIR / "frontend" / "public" / "static-api"


def write(rel: str, data) -> None:
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    with TestClient(app) as client:
        def get(route: str):
            r = client.get(route)
            r.raise_for_status()
            return r.json()

        write("chapters.json", get("/api/chapters"))
        write("stats.json", get("/api/stats"))
        everything = get("/api/exercises")
        write("all.json", everything)
        for e in everything["items"]:
            write(f"exercises/{e['id']}.json", get(f"/api/exercises/{e['id']}"))
            write(f"exercises/{e['id']}/solution.json", get(f"/api/exercises/{e['id']}/solution"))
        listing = get("/api/articles")
        write("articles.json", listing)
        for a in listing:
            write(f"articles/{a['id']}.json", get(f"/api/articles/{a['id']}"))

    con = db.open_exercises()
    docs = [dict(r) for r in con.execute(
        "SELECT s.exercise_id AS id, s.title, s.chapter, s.topics, s.prompt, s.code_terms, "
        "e.language, e.difficulty, e.source_key AS source, e.provenance "
        "FROM exercise_search s JOIN exercise e ON e.id = s.exercise_id ORDER BY s.exercise_id")]
    rows = con.execute("SELECT exercise_id, vector FROM exercise_embedding ORDER BY exercise_id").fetchall()
    if rows:   # float32, L2-normalised, 384 values per question, in question-ID order
        assert [r[0] for r in rows] == [d["id"] for d in docs], "embedding rows must match the search index"
        (OUT / "embeddings.bin").write_bytes(b"".join(r[1] for r in rows))
    write("search-index.json", {"synonyms": search.SYNONYMS, "stopwords": sorted(search.STOPWORDS), "docs": docs})
    print(f"wrote {sum(1 for _ in OUT.rglob('*.json'))} files to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
