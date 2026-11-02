"""Lesson kit. Builds article JSON for app/content/articles from verified computations.

A lesson module defines `build() -> list[Article]`. Run: app/.venv-ref/bin/python tools/lessons/build_all.py [name ...]
Every number a lesson shows must come from code in the module, and the module must assert it against the
supplied solution (`supplied(30)`) wherever that solution can run.
"""
import hashlib
import io
import itertools
import json
import math
import re
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
APP = ROOT / "app"
ARTICLES = APP / "content" / "articles"
MEDIA = APP / "media"
WORKBOOK = ROOT / "ml-interview" / "ml-coding-workbook" / "workbook.json"
_wb = None


# ---------------------------------------------------------------- the supplied solutions
def record(n: int) -> dict:
    global _wb
    if _wb is None:
        _wb = json.loads(WORKBOOK.read_text())
    return _wb[n - 1]


def solution_source(n: int) -> str:
    manifest = APP / "content" / "solutions" / f"q{n:03d}.json"
    if manifest.exists():
        source = manifest.parent / json.loads(manifest.read_text())["code_file"]
        return source.read_text()
    return record(n)["code"]


def supplied(n: int, name: str | None = None, extra: dict | None = None):
    """Run the supplied solution's code and return its main function (default `qNNN`). Used to check lesson numbers."""
    ns: dict = dict(extra or {})
    exec(compile(solution_source(n), f"<supplied q{n:03d}>", "exec"), ns)
    return ns[name or f"q{n:03d}"]


def supplied_ns(n: int, extra: dict | None = None) -> dict:
    ns: dict = dict(extra or {})
    exec(compile(solution_source(n), f"<supplied q{n:03d}>", "exec"), ns)
    return ns


def line_of(n: int, needle: str, nth: int = 1) -> int:
    """1-based line number of the nth line of solution n that contains `needle`."""
    hits = [i + 1 for i, l in enumerate(solution_source(n).split("\n")) if needle in l]
    assert len(hits) >= nth, f"{needle!r} not found {nth} time(s) in solution {n}"
    return hits[nth - 1]


def title_of(n: int) -> str:
    return record(n)["title"]


# ---------------------------------------------------------------- number formatting
def num(x, digits=4) -> str:
    if isinstance(x, (bool, np.bool_)):
        return "true" if x else "false"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    if math.isnan(x):
        return "nan"
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if abs(x - round(x)) < 1e-9 and abs(x) < 1e9:
        return str(int(round(x)))
    s = f"{x:.{digits}g}"
    return s.replace("e+0", "e").replace("e-0", "e-") if "e" in s else s


def vec(v, digits=4) -> str:
    return "(" + ", ".join(num(x, digits) for x in np.asarray(v).ravel()) + ")"


# ---------------------------------------------------------------- frame views
def cell(t, id=None, links=None, cls=None, h=None, title=None):
    d = {"t": str(t)}
    for k, v in (("id", id), ("links", links), ("cls", cls), ("h", h), ("title", title)):
        if v is not None and v != []:
            d[k] = v
    return d


def grid(cells, row_labels=None, col_labels=None, title=None, caption=None, by_sel=None):
    v = {"type": "grid", "cells": [[c if isinstance(c, dict) else {"t": str(c)} for c in row] for row in cells]}
    for k, val in (("row_labels", row_labels), ("col_labels", col_labels), ("title", title), ("caption", caption), ("by_sel", by_sel)):
        if val is not None:
            v[k] = [str(x) for x in val] if k.endswith("labels") else val
    return v


def matrix(a, name=None, id_fn=None, links_fn=None, cls_fn=None, h_fn=None, row_labels=None, col_labels=None,
           title=None, caption=None, digits=4):
    """Grid from a 1D or 2D array. id_fn(i, j), links_fn(i, j), cls_fn(i, j), h_fn(i, j) are optional."""
    a = np.asarray(a)
    if a.ndim == 1:
        a = a[None, :]
    rows = []
    for i in range(a.shape[0]):
        rows.append([cell(num(a[i, j], digits), id=id_fn(i, j) if id_fn else None, links=links_fn(i, j) if links_fn else None,
                          cls=cls_fn(i, j) if cls_fn else None, h=h_fn(i, j) if h_fn else None) for j in range(a.shape[1])])
    return grid(rows, row_labels, col_labels, title or name, caption)


def tokens(items, title=None, caption=None):
    v = {"type": "tokens", "items": [i if isinstance(i, dict) else {"t": str(i)} for i in items]}
    if title:
        v["title"] = title
    if caption:
        v["caption"] = caption
    return v


def text(t, by_sel=None):
    v = {"type": "text", "text": t}
    if by_sel:
        v["by_sel"] = by_sel
    return v


def equation(tex, by_sel=None, caption=None):
    v = {"type": "equation", "tex": tex}
    if by_sel:
        v["by_sel"] = by_sel
    if caption:
        v["caption"] = caption
    return v


def code_view(code, lines, language="Python", label=None):
    v = {"type": "code", "code": code.rstrip("\n"), "lines": lines, "language": language}
    if label:
        v["label"] = label
    return v


def series(kind, points, label, **kw):
    return {"kind": kind, "points": [[float(x), float(y)] for x, y in points], "label": label, **kw}


def plot(x, y, series_list, alt, xlabel=None, ylabel=None, title=None, caption=None, **kw):
    v = {"type": "plot", "x": [float(x[0]), float(x[1])], "y": [float(y[0]), float(y[1])], "series": series_list, "alt": alt}
    for k, val in (("xlabel", xlabel), ("ylabel", ylabel), ("title", title), ("caption", caption)):
        if val:
            v[k] = val
    v.update(kw)
    return v


def graph(nodes, edges, alt, caption=None, **kw):
    v = {"type": "graph", "nodes": nodes, "edges": edges, "alt": alt}
    if caption:
        v["caption"] = caption
    v.update(kw)
    return v


def node(id, x, y, label, sub=None, sel=None, links=None, cls=None, shape=None, title=None):
    d = {"id": id, "x": x, "y": y, "label": label}
    for k, val in (("sub", sub), ("sel", sel), ("links", links), ("cls", cls), ("shape", shape), ("title", title)):
        if val:
            d[k] = val
    return d


def edge(a, b, label=None, cls=None):
    d = {"from": a, "to": b}
    if label:
        d["label"] = label
    if cls:
        d["cls"] = cls
    return d


def frame(id, title, changed, why, fixed, *views):
    return {"id": id, "title": title, "changed": changed, "why": why, "fixed": fixed, "views": list(views)}


# ---------------------------------------------------------------- articles
class Article:
    def __init__(self, id, title, kind="lesson", prerequisites=(), exercises=(), sources=()):
        self.id, self.title, self.kind = id, title, kind
        self.prerequisites = list(prerequisites)
        self.exercises = [{"id": n, "purpose": p, "order": i + 1} for i, (n, p) in enumerate(exercises)]
        self.sources = [{"title": t, "url": u} for t, u in sources]
        self.blocks, self.demos, self.checks, self.playgrounds, self.media = [], [], [], [], []

    # -- plain blocks
    def hero(self, headline, subtitle=None):
        b = {"type": "hero", "headline": headline}
        if subtitle:
            b["subtitle"] = subtitle
        self.blocks.insert(0, b); return self

    def h2(self, t): self.blocks.append({"type": "heading", "level": 2, "text": t}); return self
    def h3(self, t): self.blocks.append({"type": "heading", "level": 3, "text": t}); return self
    def p(self, t): self.blocks.append({"type": "paragraph", "text": t}); return self
    def ul(self, items): self.blocks.append({"type": "list", "items": list(items), "ordered": False}); return self
    def ol(self, items): self.blocks.append({"type": "list", "items": list(items), "ordered": True}); return self

    def eq(self, latex, caption=None):
        b = {"type": "equation", "latex": latex}
        if caption:
            b["caption"] = caption
        self.blocks.append(b); return self

    def code(self, code, language="python", caption=None):
        b = {"type": "code", "code": textwrap.dedent(code).strip("\n"), "language": language}
        if caption:
            b["caption"] = caption
        self.blocks.append(b); return self

    def callout(self, kind, text, title=None):
        b = {"type": "callout", "kind": kind, "text": text}
        if title:
            b["title"] = title
        self.blocks.append(b); return self

    def table(self, header, rows, caption=None):
        b = {"type": "table", "header": [str(h) for h in header], "rows": [[str(c) for c in r] for r in rows]}
        if caption:
            b["caption"] = caption
        self.blocks.append(b); return self

    # -- figures (SVG saved under app/media/<article id>/)
    def figure(self, name, svg_or_fig, alt, description, caption=None):
        if hasattr(svg_or_fig, "savefig"):
            buf = io.BytesIO()
            svg_or_fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": "lesson kit"})
            svg = buf.getvalue().decode()
            import matplotlib.pyplot as plt
            plt.close(svg_or_fig)
        else:
            svg = svg_or_fig
        rel = f"{self.id}/{name}.svg"
        path = MEDIA / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text() != svg:
            path.write_text(svg)
        self.media.append({"path": rel, "description": description, "alt_text": alt, "attribution": "Figure generated for this site."})
        b = {"type": "image", "path": rel}
        if caption:
            b["caption"] = caption
        self.blocks.append(b); return self

    def figure_file(self, name, alt, description, caption=None, optional=True):
        """An image rendered earlier by app/e2e/render_figures.mjs (PNG in app/media/<article id>/). Skipped until it exists."""
        rel = f"{self.id}/{name}.png"
        if not (MEDIA / rel).is_file():
            assert optional, f"missing figure {rel}: run app/e2e/render_figures.mjs first"
            print(f"  note: {rel} not rendered yet; run app/e2e/render_figures.mjs, then build again")
            return self
        self.media.append({"path": rel, "description": description, "alt_text": alt, "attribution": "Figure generated for this site."})
        b = {"type": "image", "path": rel}
        if caption:
            b["caption"] = caption
        self.blocks.append(b); return self

    def gallery(self, items, optional=True):
        """items: [(name, alt, description, caption)] of PNGs rendered earlier. Shown as a grid."""
        got = []
        for name, alt, description, caption in items:
            rel = f"{self.id}/{name}.png"
            if not (MEDIA / rel).is_file():
                assert optional, f"missing figure {rel}"
                print(f"  note: {rel} not rendered yet")
                continue
            self.media.append({"path": rel, "description": description, "alt_text": alt, "attribution": "Figure generated for this site."})
            got.append({"path": rel, "caption": caption})
        if got:
            self.blocks.append({"type": "gallery", "items": got})
        return self

    # -- demonstrations
    def demo(self, id, caption, frames_fn, params=(), help=(), facts=(), selection_links=()):
        """frames_fn(**param_values) -> list of frames. params: [(name, label, [(value, label), ...])]."""
        plist = [{"name": n, "label": lab, "options": [{"value": v, "label": l} for v, l in opts]} for n, lab, opts in params]
        cases, order = {}, {}
        combos = itertools.product(*[[v for v, _ in opts] for _, _, opts in params]) if params else [()]
        for combo in combos:
            vals = {p[0]: v for p, v in zip(params, combo)}
            frames = frames_fn(**vals)
            cases["|".join(str(v) for v in combo)] = {"frames": frames}
            for f in frames:
                order.setdefault(f["id"], f["title"])
        defaults = {n: opts[0][0] if default_first(opts) else opts[0][0] for n, _, opts in params}
        self.demos.append({
            "id": id, "component": "concept-lab", "seed": None,
            "input_schema": {"defaults": defaults},
            "steps": [{"id": k, "title": t} for k, t in order.items()],
            "data": {"params": plist, "cases": cases},
            "expected": [{"claim": c, "verified_against": s} for c, s in facts],
            "selection_links": list(selection_links),
            "help": list(help),
        })
        self.blocks.append({"type": "demonstration", "id": id, "caption": caption}); return self

    def scene_demo(self, id, caption, scene_fn, params=(), help=(), facts=()):
        """scene_fn(**param_values) -> scene.Scene. Produces a `scene` demonstration (3D timeline)."""
        plist = [{"name": n, "label": lab, "options": [{"value": v, "label": l} for v, l in opts]} for n, lab, opts in params]
        cases, order = {}, {}
        combos = itertools.product(*[[v for v, _ in opts] for _, _, opts in params]) if params else [()]
        for combo in combos:
            vals = {p[0]: v for p, v in zip(params, combo)}
            sc = scene_fn(**vals)
            cases["|".join(str(v) for v in combo)] = {"scene": sc.to_dict()}
            for c in sc.chapters:
                order.setdefault(c["id"], c["title"])
        self.demos.append({
            "id": id, "component": "scene", "seed": None,
            "input_schema": {"defaults": {n: opts[0][0] for n, _, opts in params}},
            "steps": [{"id": k, "title": t} for k, t in order.items()],
            "data": {"params": plist, "cases": cases},
            "expected": [{"claim": c, "verified_against": s} for c, s in facts],
            "help": list(help),
        })
        self.blocks.append({"type": "demonstration", "id": id, "caption": caption}); return self

    def demo_raw(self, demo, caption):
        """Add a demonstration for a hand-written component such as broadcast-trace."""
        self.demos.append(demo)
        self.blocks.append({"type": "demonstration", "id": demo["id"], "caption": caption}); return self

    def set_default(self, demo_id, **values):
        for d in self.demos:
            if d["id"] == demo_id:
                d["input_schema"]["defaults"].update(values)
        return self

    def compare(self, demo_id, changed, cases, takeaway, frame=None):
        b = {"type": "compare", "demonstration": demo_id, "changed": changed,
             "cases": [{"label": l, "inputs": i} for l, i in cases], "takeaway": takeaway}
        if frame:
            b["frame"] = frame
        self.blocks.append(b); return self

    # -- checks
    def check(self, id, kind, prompt, answer_type, answer=None, explanation=None, options=None, case=None, tolerance=None, demonstration=None):
        c = {"id": id, "kind": kind, "prompt": prompt, "answer_type": answer_type, "explanation": explanation or {}}
        if answer is not None:
            c["answer"] = answer
        for k, v in (("options", options), ("case", case), ("tolerance", tolerance), ("demonstration", demonstration)):
            if v is not None:
                c[k] = v
        self.checks.append(c)
        self.blocks.append({"type": "check", "id": id}); return self

    # -- solution walkthrough (verified against the real solution when the article is imported)
    def walkthrough(self, exercise, notes, title=None):
        lines = solution_source(exercise).split("\n")
        out = []
        for ln, match, txt in notes:
            lo, hi = (ln, ln) if isinstance(ln, int) else ln
            assert match in "\n".join(lines[lo - 1:hi]), f"walkthrough {exercise}: {match!r} not in lines {lo}-{hi}"
            out.append({"lines": [lo, hi], "match": match, "text": txt})
        b = {"type": "walkthrough", "exercise_id": exercise, "notes": out}
        if title:
            b["title"] = title
        self.blocks.append(b); return self

    def playground(self, exercise, starter, checks=(), note=None, packages=("numpy",), time_limit_ms=10000,
                   max_input_size=None, runtime="pyodide"):
        p = {"exercise_id": exercise, "runtime": runtime, "time_limit_ms": time_limit_ms}
        if runtime == "pyodide":
            p.update(starter_code=textwrap.dedent(starter).strip("\n") + "\n", packages=list(packages),
                     checks=[{"name": n, "code": textwrap.dedent(c).strip("\n")} for n, c in checks])
        elif starter:
            p["starter_code"] = textwrap.dedent(starter).strip("\n") + "\n"
        if note:
            p["note"] = note
        if max_input_size:
            p["max_input_size"] = max_input_size
        self.playgrounds.append(p)
        self.blocks.append({"type": "playground", "exercise_id": exercise}); return self

    # -- output
    def to_dict(self, review_state="complete"):
        d = {"id": self.id, "kind": self.kind, "title": self.title, "revision": 1, "review_state": review_state,
             "blocks": self.blocks}
        for k, v in (("prerequisites", self.prerequisites), ("exercises", self.exercises), ("sources", self.sources),
                     ("media", self.media), ("demonstrations", self.demos), ("checks", self.checks), ("playgrounds", self.playgrounds)):
            if v:
                d[k] = v
        return d


def default_first(opts):
    return True


def _hash(d):
    doc = {k: v for k, v in d.items() if k != "revision"}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write(article: Article, review_state="complete") -> str:
    """Write the article JSON. The revision goes up by one when the content changed."""
    d = article.to_dict(review_state)
    path = ARTICLES / f"{article.id}.json"
    ARTICLES.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = json.loads(path.read_text())
        if _hash(old) == _hash(d):
            return "unchanged"
        d["revision"] = old["revision"] + 1
    path.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
    return f"written (revision {d['revision']})"


# ---------------------------------------------------------------- explanation helpers
def wrong(*pairs):
    """Build the `wrong` map of an explanation: wrong(("(3, 1)", "That is shape A."), ...)."""
    return {k: v for k, v in pairs}


def expl(correct, default, **wrong_map):
    d = {"correct": correct, "default": default}
    if wrong_map:
        d["wrong"] = wrong_map
    return d
