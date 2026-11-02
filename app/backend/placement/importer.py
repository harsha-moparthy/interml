"""Build exercises.db from the audited sources. Usage: python -m placement.importer [--allow-drift]

Reads original prompts/code from workbook.json and applies maintained solution overrides.
PDF text is used for page ranges, never for code.
Writes a temporary database, checks it, then replaces exercises.db. Never touches content.db.
"""
import argparse
import ast
import hashlib
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .curated import load_solutions
from .reading_groups import overrides as reading_overrides
from .resources import answer_guide, select_readings

REPO = config.APP_DIR.parent
SOURCES = {
    "pdf": "ml-interview/ml-coding-patterns-500-solutions.pdf",
    "tex": "ml-interview/ml-coding-patterns-500-solutions.tex",
    "workbook": "ml-interview/ml-coding-workbook/workbook.json",
    "catalog": "ml-interview/ml-coding-catalog/catalog.json",
    "supplemental": "addl_questions.md",
}
AUDIT = REPO / "planning" / "source-audit.json"
READING_OVERRIDES = reading_overrides()
LICENSE_NOTE = (
    "Derived from a repository covered by the Educational Use Only License "
    "(ml-interview/ml-coding-workbook/LICENSE-Deep-ML.txt): no commercial use, no model training."
)


def _split_identifier(name: str) -> list[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).replace("_", " ")
    return [w.lower() for w in spaced.split() if len(w) > 1 and not w.isdigit()]


def code_terms(code: str, language: str, number: int) -> str:
    """Words from the solution that name the concepts used: imports, called functions, comments, docstrings.
    Used only for search. The solution itself is never shown by search."""
    words: list[str] = []
    if language == "Python":
        try:
            tree = ast.parse(code)
        except SyntaxError:
            tree = None
        if tree:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    words += [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    words.append(node.module.split(".")[0])
                elif isinstance(node, ast.Attribute):
                    words += _split_identifier(node.attr)
                elif isinstance(node, ast.FunctionDef) and node.name != f"q{number:03d}":
                    words += _split_identifier(node.name)
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    words += _split_identifier(node.func.id)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    doc = ast.get_docstring(node)
                    if doc:
                        words += re.findall(r"[A-Za-z]{3,}", doc)
    else:
        words += re.findall(r"__(?:global|shared|syncthreads|device|host)__|\b(?:cuda|template|thread|block|warp)\w*", code)
    words += re.findall(r"#\s*(.+)", code)
    seen, out = set(), []
    for w in " ".join(words).split():
        w = w.lower().strip("(),.:;")
        if len(w) > 2 and w not in seen and w not in {"self", "the", "and", "for", "return", "import", "def"}:
            seen.add(w)
            out.append(w)
    return " ".join(out)


def embedding_text(r: dict, chapter: str, subsection: str, terms: str) -> str:
    return f"{r['title']}. {chapter}. {subsection}. {r['question']} Concepts: {terms}"


class ImportError_(Exception):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_sources(allow_drift: bool) -> dict:
    audit = json.loads(AUDIT.read_text())
    hashes = {}
    for name, rel in SOURCES.items():
        hashes[name] = sha256(REPO / rel)
        if name in audit["hashes"] and hashes[name] != audit["hashes"][name] and not allow_drift:
            raise ImportError_(f"source drift: {rel} hash differs from planning/source-audit.json "
                               "(re-run tools/audit_source.py, then pass --allow-drift)")
    return {"hashes": hashes, "pages": {r["id"]: r["pdf_pages"] for r in audit["records"]}}


def load_records():
    workbook = json.loads((REPO / SOURCES["workbook"]).read_text())
    catalog = json.loads((REPO / SOURCES["catalog"]).read_text())
    cat_items = {i["number"]: i for s in catalog for sub in s["subsections"] for i in sub["items"]}
    return workbook, catalog, cat_items


def _plain_title(prompt: str) -> str:
    """Make a concise display title while keeping the full Markdown in prompt."""
    bold = re.findall(r"\*\*([^*]+)\*\*", prompt)
    if bold:
        title = " · ".join(dict.fromkeys(bold[:3]))
    else:
        title = re.split(r"[?:]", prompt, 1)[0]
    title = re.sub(r"[`*_]", "", title)
    title = re.sub(r"\$([^$]+)\$", r"\1", title)
    title = re.sub(r"\s+", " ", title).strip(" .")
    return title[:120] + ("…" if len(title) > 120 else "")


def load_supplemental() -> tuple[list[dict], list[dict]]:
    """Parse addl_questions.md into 18 chapters and exactly 500 records."""
    lines = (REPO / SOURCES["supplemental"]).read_text().splitlines()
    chapters, records = [], []
    chapter = subsection = None
    current = None

    def finish():
        nonlocal current
        if not current:
            return
        prompt = "\n".join(current["lines"]).strip()
        current["prompt"] = prompt
        current["title"] = _plain_title(prompt)
        del current["lines"]
        records.append(current)
        current = None

    for line in lines:
        m = re.match(r"^## Section (\d+):\s*(.+)$", line)
        if m:
            finish()
            chapter = {"number": int(m.group(1)), "title": m.group(2).strip(), "subsections": []}
            chapters.append(chapter)
            subsection = None
            continue
        m = re.match(r"^###\s+([A-Z])\.\s*(.+)$", line)
        if m:
            finish()
            subsection = {"position": len(chapter["subsections"]) + 1, "title": m.group(2).strip()}
            chapter["subsections"].append(subsection)
            continue
        m = re.match(r"^(\d+)\.\s+(.*)$", line)
        if m:
            finish()
            if chapter is None:
                raise ImportError_("supplemental question appears before a Section heading")
            if subsection is None:
                subsection = {"position": len(chapter["subsections"]) + 1, "title": "General"}
                chapter["subsections"].append(subsection)
            source_number = int(m.group(1))
            current = {"source_number": source_number, "id": 600 + source_number,
                       "chapter_number": chapter["number"], "chapter_title": chapter["title"],
                       "subsection_position": subsection["position"], "subsection_title": subsection["title"],
                       "lines": [m.group(2)]}
            continue
        if current is not None and line.strip() not in {"---"}:
            current["lines"].append(line)
    finish()
    ids = [r["source_number"] for r in records]
    if ids != list(range(1, 501)):
        raise ImportError_("addl_questions.md must contain question numbers 1..500 exactly once and in order")
    if [c["number"] for c in chapters] != list(range(1, 19)):
        raise ImportError_("addl_questions.md must contain Section 1..18 exactly once and in order")
    return records, chapters


def validate(workbook, catalog, cat_items, pages):
    errors = []
    ids = [r["number"] for r in workbook]
    if len(set(ids)) != len(ids):
        errors.append(f"duplicate question IDs: {sorted({i for i in ids if ids.count(i) > 1})}")
    if ids != list(range(1, 601)):
        errors.append("workbook IDs are not exactly 1..600 in order")
    if sorted(cat_items) != list(range(1, 601)):
        errors.append("catalog does not cover exactly IDs 1..600")
    for r in workbook:
        n = r["number"]
        if not (r.get("code") or "").strip():
            errors.append(f"{n:03d}: missing solution")
        if not (r.get("question") or "").strip():
            errors.append(f"{n:03d}: missing prompt")
        if not pages.get(n):
            errors.append(f"{n:03d}: missing PDF page range")
        c = cat_items.get(n)
        if c and r["title"] not in (c["title"], c["display_title"]):
            errors.append(f"{n:03d}: title differs between workbook and catalog")
    return errors


def build(db_path: Path, workbook, catalog, cat_items, supplemental, supplemental_chapters, info):
    curated = load_solutions()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    con = sqlite3.connect(db_path)
    con.executescript((config.SCHEMA_DIR / "exercises.sql").read_text())
    for name, h in info["hashes"].items():
        con.execute("INSERT INTO source_file VALUES (?,?,?,?)", (name, SOURCES[name], h, now))
    sub_ids = {}
    for sec in catalog:
        cid = f"C{sec['number']:02d}"
        con.execute("INSERT INTO chapter VALUES (?,?,?)", (cid, sec["number"], sec["title"]))
        for pos, sub in enumerate(sec["subsections"], 1):
            cur = con.execute("INSERT INTO subsection (chapter_id, position, title) VALUES (?,?,?)",
                              (cid, pos, sub["title"]))
            for item in sub["items"]:
                sub_ids[item["number"]] = (cid, cur.lastrowid, sub["title"], sec["title"])
    supplemental_sub_ids = {}
    for sec in supplemental_chapters:
        cid = f"A{sec['number']:02d}"
        con.execute("INSERT INTO chapter VALUES (?,?,?)", (cid, 36 + sec["number"], sec["title"]))
        for sub in sec["subsections"]:
            cur = con.execute("INSERT INTO subsection (chapter_id, position, title) VALUES (?,?,?)",
                              (cid, sub["position"], sub["title"]))
            supplemental_sub_ids[(sec["number"], sub["position"])] = cur.lastrowid
    docs: list[tuple[int, str]] = []
    for r in workbook:
        n = r["number"]
        source_code = r["code"]
        r = dict(r)  # Keep the source workbook unchanged.
        reviewed = curated.get(n, {})
        r["code"] = reviewed.get("code") or r["code"]
        cid, sid, sub_title, chap_title = sub_ids[n]
        c = cat_items[n]
        first, last = info["pages"][n]
        repo = r["provenance"] == "repository"
        rec_hash = hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
        con.execute(
            "INSERT INTO exercise (id, list_position, chapter_id, subsection_id, title, prompt, solution_code,"
            " answer_text, content_kind, supplemental, language, provenance, source_key, source_url, repository_url, difficulty, license_note,"
            " pdf_page_first, pdf_page_last, record_sha256) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (n, n, cid, sid, r["title"], r["question"], r["code"], reviewed.get("answer", ""), "coding", 0, r["language"], r["provenance"],
             r["source"], r["url"], r["repository_url"], c.get("difficulty"),
             LICENSE_NOTE if repo else None, first, last, rec_hash))
        topics = {sub_title, r.get("subsection") or "", c.get("category") or "", c.get("subsection") or ""}
        topics = sorted(t for t in topics if t)
        con.executemany("INSERT INTO exercise_topic VALUES (?,?)", [(n, t) for t in topics])
        # Presentation-only documentation changes must not perturb search ranking.
        terms = code_terms(source_code, r["language"], n)
        con.execute("INSERT INTO exercise_search (exercise_id, id_text, title, chapter, topics, prompt, code_terms) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (n, f"{n:03d} {n}", r["title"], chap_title, " ".join(topics), r["question"], terms))
        docs.append((n, embedding_text(r, chap_title, sub_title, terms)))
        # Do not use implementation identifiers for reading selection. A helper
        # such as np.clip can otherwise look like the CLIP model, for example.
        chosen = reviewed.get("readings") or READING_OVERRIDES.get(n) or select_readings(f"{r['title']} {r['question']}", f"{chap_title} {sub_title}")
        for rank, reading in enumerate(chosen, 1):
            con.execute("INSERT INTO exercise_resource VALUES (?,?,?,?,?,?,?)",
                        (n, rank, reading.title, reading.url, reading.publisher, reading.access, reading.why))

    for r in supplemental:
        n = r["id"]
        cid = f"A{r['chapter_number']:02d}"
        sid = supplemental_sub_ids[(r["chapter_number"], r["subsection_position"])]
        if r["source_number"] <= 52:
            kind, language = "coding", "Python"
        elif r["chapter_number"] == 2 or r["prompt"].lstrip().lower().startswith("design "):
            kind, language = "system-design", "Architecture"
        else:
            kind, language = "theory", "Theory"
        reviewed = curated.get(n, {})
        readings = reviewed.get("readings") or READING_OVERRIDES.get(n) or select_readings(f"{r['title']} {r['prompt']}", f"{r['chapter_title']} {r['subsection_title']}")
        answer = reviewed.get("answer") or answer_guide(r["prompt"], readings, kind)
        code = reviewed.get("code", "")
        rec_hash = hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
        con.execute(
            "INSERT INTO exercise (id, list_position, chapter_id, subsection_id, title, prompt, solution_code, answer_text,"
            " content_kind, supplemental, language, provenance, source_key, difficulty, record_sha256) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (n, n, cid, sid, r["title"], r["prompt"], code, answer, kind, 1, language,
             "supplemental", "ADDL", "Advanced", rec_hash))
        topics = sorted({r["subsection_title"], r["chapter_title"]})
        con.executemany("INSERT INTO exercise_topic VALUES (?,?)", [(n, t) for t in topics])
        con.execute("INSERT INTO exercise_search (exercise_id, id_text, title, chapter, topics, prompt, code_terms) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (n, f"{n:03d} {n} supplemental {r['source_number']}", r["title"], r["chapter_title"],
                     " ".join(topics), r["prompt"], ""))
        docs.append((n, f"{r['title']}. {r['chapter_title']}. {r['subsection_title']}. {r['prompt']}"))
        for rank, reading in enumerate(readings, 1):
            con.execute("INSERT INTO exercise_resource VALUES (?,?,?,?,?,?,?)",
                        (n, rank, reading.title, reading.url, reading.publisher, reading.access, reading.why))
    con.execute("INSERT INTO exercise_search(exercise_search) VALUES ('optimize')")
    add_embeddings(con, docs)
    con.commit()
    con.close()


def add_embeddings(con: sqlite3.Connection, docs: list[tuple[int, str]]) -> None:
    """Store a vector per question when the local model is installed; otherwise record that it is off."""
    from . import semantic
    if not semantic.available():
        con.execute("INSERT INTO meta VALUES ('embedding_status', 'off: model not installed (python -m placement.semantic download)')")
        print("note: semantic model not installed; semantic search will be off", file=sys.stderr)
        return
    vecs = semantic.embed_documents([t for _, t in docs])
    con.executemany("INSERT INTO exercise_embedding VALUES (?, ?)",
                    [(n, vecs[i].astype("float32").tobytes()) for i, (n, _) in enumerate(docs)])
    con.execute("INSERT INTO meta VALUES ('embedding_model', ?)", (semantic.MODEL_NAME,))
    con.execute("INSERT INTO meta VALUES ('embedding_status', 'on')")


def verify(db_path: Path):
    con = sqlite3.connect(db_path)
    problems = []
    if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
        problems.append("integrity_check failed")
    if con.execute("PRAGMA foreign_key_check").fetchall():
        problems.append("foreign key violations")
    ids = [r[0] for r in con.execute("SELECT id FROM exercise ORDER BY list_position")]
    if ids != list(range(1, 1101)):
        problems.append("exercise IDs are not exactly 1..1100 in original order")
    if con.execute("SELECT count(*) FROM chapter").fetchone()[0] != 54:
        problems.append("chapter count is not 54")
    if con.execute("SELECT count(*) FROM exercise_search").fetchone()[0] != 1100:
        problems.append("search index does not hold 1100 rows")
    invalid_resources = con.execute(
        "SELECT e.id FROM exercise e LEFT JOIN exercise_resource r ON r.exercise_id=e.id "
        "GROUP BY e.id HAVING count(r.rank) != CASE WHEN e.supplemental=1 THEN 5 ELSE 2 END"
    ).fetchall()
    if invalid_resources:
        problems.append("expected 2 workbook readings and 5 appendix readings per question")
    emb = con.execute("SELECT count(*) FROM exercise_embedding").fetchone()[0]
    status = con.execute("SELECT value FROM meta WHERE key='embedding_status'").fetchone()[0]
    if status == "on" and emb != 1100:
        problems.append(f"embeddings: {emb} vectors, expected 1100")
    if con.execute("SELECT count(*) FROM exercise WHERE solution_code = '' AND answer_text = ''").fetchone()[0]:
        problems.append("question without a solution or answer guide")
    con.close()
    return problems


def run(allow_drift=False):
    info = check_sources(allow_drift)
    workbook, catalog, cat_items = load_records()
    supplemental, supplemental_chapters = load_supplemental()
    errors = validate(workbook, catalog, cat_items, info["pages"])
    if errors:
        raise ImportError_("\n".join(errors))
    config.DATA_DIR.mkdir(exist_ok=True)
    tmp = config.EXERCISES_DB.with_suffix(".db.tmp")
    tmp.unlink(missing_ok=True)
    try:
        build(tmp, workbook, catalog, cat_items, supplemental, supplemental_chapters, info)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    problems = verify(tmp)
    if problems:
        tmp.unlink()
        raise ImportError_("\n".join(problems))
    os.replace(tmp, config.EXERCISES_DB)
    return len(workbook) + len(supplemental)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-drift", action="store_true")
    try:
        print(f"imported {run(ap.parse_args().allow_drift)} exercises -> {config.EXERCISES_DB}")
    except ImportError_ as e:
        print(f"IMPORT FAILED\n{e}", file=sys.stderr)
        sys.exit(1)
