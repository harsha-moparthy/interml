import json
import re
import sqlite3
from fastapi import APIRouter, HTTPException, Query

from . import config, db, search as search_engine

router = APIRouter(prefix="/api")


def selected_ids() -> dict[int, str]:
    path = config.APP_DIR / "content" / "lesson_selection.json"  # always the repository copy
    return {s["exercise_id"]: s["chapter"] for s in json.loads(path.read_text())["selected"]}


def completed_lessons() -> set[int]:
    con = db.open_content()
    try:
        rows = con.execute(
            "SELECT ea.exercise_id FROM exercise_article ea JOIN article a ON a.id = ea.article_id "
            "WHERE ea.purpose = 'deep-lesson' AND a.kind = 'lesson' AND a.review_state = 'complete'")
        return {r[0] for r in rows}
    finally:
        con.close()


def fts_query(q: str) -> str | None:
    tokens = re.findall(r"[A-Za-z0-9]+", q)
    return " AND ".join(f'"{t}"*' for t in tokens) if tokens else None


def lesson_status(eid: int, planned: dict, done: set) -> str:
    return "complete" if eid in done else "planned" if eid in planned else "none"


@router.get("/search")
def search_questions(q: str = "", chapter: str = "", lesson: str = Query("", pattern="^(|complete|planned|none)$"),
                     limit: int = Query(40, ge=1, le=1100), mode: str = Query("hybrid", pattern="^(hybrid|keyword|semantic)$")):
    """Ranked search. See placement/search.py for the query syntax."""
    planned, done = selected_ids(), completed_lessons()
    status = lambda i: lesson_status(i, planned, done)
    if chapter:
        q += f" chapter:{chapter}"
    if lesson:
        q += f" lesson:{lesson}"
    result = search_engine.search(q, lesson_status=status, mode=mode, limit=limit)
    con = db.open_exercises()
    ids = result["ids"]
    rows = {}
    if ids:
        marks = ",".join("?" * len(ids))
        for r in con.execute(
                "SELECT e.id, e.title, e.prompt, e.chapter_id, c.title AS chapter_title, s.title AS subsection_title, "
                f"e.difficulty, e.language FROM exercise e JOIN chapter c ON c.id = e.chapter_id "
                f"JOIN subsection s ON s.id = e.subsection_id WHERE e.id IN ({marks})", ids):
            rows[r["id"]] = dict(r)
    items = search_engine.present(result, rows)
    for it in items:
        it["lesson"] = status(it["id"])
    return {"query": q, "total": result["total"], "items": items, "corrected": result["corrected"],
            "filters": result["filters"], "semantic": result["semantic"], "mode": mode}


@router.get("/stats")
def stats():
    con = db.open_exercises()
    imported = con.execute("SELECT count(*) FROM exercise").fetchone()[0]
    return {"imported_questions": imported, "planned_deep_lessons": len(selected_ids()),
            "completed_deep_lessons": len(completed_lessons() & set(selected_ids()))}


@router.get("/chapters")
def chapters():
    con = db.open_exercises()
    rows = con.execute(
        "SELECT c.id, c.position, c.title, count(e.id) AS count, min(e.id) AS first_id, max(e.id) AS last_id "
        "FROM chapter c JOIN exercise e ON e.chapter_id = c.id GROUP BY c.id ORDER BY c.position")
    return [dict(r) for r in rows]


@router.get("/exercises")
def exercises(q: str = "", chapter: str = "", lesson: str = Query("", pattern="^(|complete|planned|none)$"),
              limit: int = Query(1100, ge=1, le=1100), offset: int = Query(0, ge=0)):
    con = db.open_exercises()
    where, args = [], []
    if q.strip():
        fq = fts_query(q)
        if fq is None:
            return {"total": 0, "items": []}
        where.append("e.id IN (SELECT exercise_id FROM exercise_search WHERE exercise_search MATCH ?)")
        args.append(fq)
    if chapter:
        where.append("e.chapter_id = ?")
        args.append(chapter)
    sql = ("SELECT e.id, e.title, e.chapter_id, c.title AS chapter_title, s.title AS subsection_title, "
           "e.difficulty, e.language FROM exercise e JOIN chapter c ON c.id = e.chapter_id "
           "JOIN subsection s ON s.id = e.subsection_id")
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY e.list_position"
    planned, done = selected_ids(), completed_lessons()
    items = []
    for r in con.execute(sql, args):
        d = dict(r)
        d["lesson"] = lesson_status(d["id"], planned, done)
        if not lesson or d["lesson"] == lesson:
            items.append(d)
    return {"total": len(items), "items": items[offset:offset + limit]}


def _exercise_row(con, eid: int, columns: str):
    row = con.execute(f"SELECT {columns} FROM exercise e JOIN chapter c ON c.id = e.chapter_id "
                      "JOIN subsection s ON s.id = e.subsection_id WHERE e.id = ?", (eid,)).fetchone()
    if row is None:
        raise HTTPException(404, f"No question {eid}")
    return row


@router.get("/exercises/{eid}")
def exercise(eid: int):
    """The original task and its metadata. The solution is a separate route and is not sent here."""
    con = db.open_exercises()
    row = _exercise_row(
        con, eid,
        "e.id, e.title, e.prompt, e.language, e.provenance, e.source_key, e.source_url, e.repository_url, "
        "e.difficulty, e.license_note, e.pdf_page_first, e.pdf_page_last, e.chapter_id, "
        "e.content_kind, e.supplemental, "
        "CASE WHEN e.solution_code != '' THEN 'solution' "
        "WHEN e.answer_text LIKE 'Answer focus:%' THEN 'guide' ELSE 'answer' END AS answer_type, "
        "c.title AS chapter_title, s.title AS subsection_title")
    d = dict(row)
    d["lesson"] = lesson_status(eid, selected_ids(), completed_lessons())
    d["resources"] = [dict(r) for r in con.execute(
        "SELECT rank, title, url, publisher, access_note, why FROM exercise_resource "
        "WHERE exercise_id = ? ORDER BY rank", (eid,))]
    c = db.open_content()
    try:
        d["articles"] = [dict(r) for r in c.execute(
            "SELECT a.id, a.title, a.kind, a.review_state, ea.purpose FROM exercise_article ea "
            "JOIN article a ON a.id = ea.article_id WHERE ea.exercise_id = ? ORDER BY ea.reading_order", (eid,))]
    finally:
        c.close()
    d["previous_id"] = eid - 1 if eid > 1 else None
    d["next_id"] = eid + 1 if eid < 1100 else None
    return d


@router.get("/exercises/{eid}/solution")
def solution(eid: int):
    con = db.open_exercises()
    row = _exercise_row(con, eid, "e.id, e.solution_code, e.answer_text, e.content_kind, e.language")
    return {"id": row["id"], "language": row["language"], "code": row["solution_code"],
            "answer": row["answer_text"], "content_kind": row["content_kind"]}


@router.get("/articles")
def articles():
    con = db.open_content()
    try:
        return [dict(r) for r in con.execute("SELECT id, kind, title, review_state, revision FROM article ORDER BY title")]
    finally:
        con.close()


@router.get("/articles/{aid}")
def article(aid: str):
    con = db.open_content()
    try:
        a = con.execute("SELECT id, kind, title, body, revision, review_state, sources FROM article WHERE id = ?",
                        (aid,)).fetchone()
        if a is None:
            raise HTTPException(404, f"No article {aid}")
        d = dict(a)
        d["blocks"] = json.loads(d.pop("body"))
        d["sources"] = json.loads(d["sources"])
        d["prerequisites"] = [dict(r) for r in con.execute(
            "SELECT a.id, a.title, a.review_state FROM article_prerequisite p JOIN article a ON a.id = p.requires_id "
            "WHERE p.article_id = ? ORDER BY a.title", (aid,))]
        d["required_by"] = [dict(r) for r in con.execute(
            "SELECT a.id, a.title FROM article_prerequisite p JOIN article a ON a.id = p.article_id "
            "WHERE p.requires_id = ? ORDER BY a.title", (aid,))]
        d["media"] = {r["path"]: {"alt_text": r["alt_text"], "description": r["description"], "attribution": r["attribution"]}
                      for r in con.execute("SELECT m.* FROM media m JOIN article_media am ON am.media_id = m.id "
                                           "WHERE am.article_id = ?", (aid,))}
        d["demonstrations"] = {}
        for r in con.execute("SELECT * FROM demonstration WHERE article_id = ?", (aid,)):
            help_rows = con.execute("SELECT h.step_id, h.help_type, h.content, h.prerequisite_id, a.title AS prerequisite_title "
                                    "FROM step_help h LEFT JOIN article a ON a.id = h.prerequisite_id "
                                    "WHERE h.demonstration_id = ? ORDER BY h.id", (r["id"],))
            d["demonstrations"][r["id"]] = {
                "component": r["component"], "input_schema": json.loads(r["input_schema"]), "seed": r["seed"],
                "steps": json.loads(r["steps"]), "selection_links": json.loads(r["selection_links"]),
                "data": json.loads(r["data"]) if r["data"] else None,
                "help": [dict(h) for h in help_rows]}
        d["checks"] = {}
        for r in con.execute("SELECT * FROM learning_check WHERE article_id = ?", (aid,)):
            d["checks"][r["check_id"]] = {"kind": r["kind"], **json.loads(r["inputs"]), "answer": json.loads(r["answer"]),
                                          "explanation": json.loads(r["explanation"])}
        d["playgrounds"] = {}
        for r in con.execute("SELECT * FROM playground WHERE exercise_id IN "
                             "(SELECT exercise_id FROM exercise_article WHERE article_id = ?)", (aid,)):
            d["playgrounds"][r["exercise_id"]] = {
                "runtime": r["runtime"], "packages": json.loads(r["packages"]), "starter_code": r["starter_code"],
                "checks": json.loads(r["checks"] or "[]"), "time_limit_ms": r["time_limit_ms"],
                "max_input_size": r["max_input_size"], "note": r["note"]}
        ex_ids = [r["exercise_id"] for r in con.execute(
            "SELECT exercise_id FROM exercise_article WHERE article_id = ? ORDER BY reading_order", (aid,))]
    finally:
        con.close()
    ex = db.open_exercises()
    d["exercises"] = [dict(r) for i in ex_ids for r in ex.execute("SELECT id, title FROM exercise WHERE id = ?", (i,))]
    return d
