"""Cross-database and content checks. Usage: python -m placement.check_links
Checks: exercise_id references, prerequisite cycles, missing media files, article references."""
import sqlite3
import sys

from . import config


def find_cycle(edges: dict[str, list[str]]):
    state, stack = {}, []

    def visit(n):
        state[n] = 1
        stack.append(n)
        for m in edges.get(n, []):
            if state.get(m) == 1:
                return stack[stack.index(m):] + [m]
            if m not in state and (c := visit(m)):
                return c
        state[n] = 2
        stack.pop()

    for n in list(edges):
        if n not in state and (c := visit(n)):
            return c
    return None


def check(exercises_path=None, content_path=None, media_dir=None) -> list[str]:
    ex = sqlite3.connect(f"file:{exercises_path or config.EXERCISES_DB}?mode=ro", uri=True)
    ct = sqlite3.connect(content_path or config.CONTENT_DB)
    media_dir = media_dir or config.MEDIA_DIR
    problems = []
    known = {r[0] for r in ex.execute("SELECT id FROM exercise")}
    for table in ("exercise_article", "playground"):
        for (eid,) in ct.execute(f"SELECT exercise_id FROM {table}"):
            if eid not in known:
                problems.append(f"{table}: unknown exercise_id {eid}")
    articles = {r[0] for r in ct.execute("SELECT id FROM article")}
    for aid, req in ct.execute("SELECT article_id, requires_id FROM article_prerequisite"):
        for x in (aid, req):
            if x not in articles:
                problems.append(f"prerequisite references missing article {x}")
    edges: dict[str, list[str]] = {}
    for aid, req in ct.execute("SELECT article_id, requires_id FROM article_prerequisite"):
        edges.setdefault(aid, []).append(req)
    if cycle := find_cycle(edges):
        problems.append("prerequisite cycle: " + " -> ".join(cycle))
    for (path,) in ct.execute("SELECT path FROM media"):
        if path.startswith("/") or ".." in path.split("/"):
            problems.append(f"media path is not relative to the app media folder: {path}")
        elif not (media_dir / path).is_file():
            problems.append(f"media file missing: {path}")
    return problems


if __name__ == "__main__":
    found = check()
    print("\n".join(found) if found else "links ok")
    sys.exit(1 if found else 0)
