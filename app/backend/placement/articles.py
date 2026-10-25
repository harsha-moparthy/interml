"""Import authored article sources into content.db. Usage: python -m placement.articles [check]

Sources are JSON files under CONTENT_DIR/articles (see planning/CONTENT_FORMAT.md).
The import validates every file first, then writes all changes in one transaction.
It never deletes articles and never lowers a revision.
"""
import hashlib
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import check_links, config, db

BLOCK_FIELDS = {
    "heading": {"level", "text"},
    "paragraph": {"text"},
    "list": {"items", "ordered"},
    "equation": {"latex", "caption"},
    "code": {"language", "code", "caption"},
    "image": {"path", "caption"},
    "callout": {"kind", "title", "text"},
    "table": {"header", "rows", "caption"},
    "demonstration": {"id", "caption"},
    "check": {"id"},
    "compare": {"demonstration", "changed", "cases", "takeaway", "frame"},
    "playground": {"exercise_id"},
    "walkthrough": {"exercise_id", "title", "notes"},
    "hero": {"headline", "subtitle"},
    "gallery": {"items", "caption"},
}
REQUIRED = {
    "heading": {"level", "text"}, "paragraph": {"text"}, "list": {"items"}, "equation": {"latex"},
    "code": {"code"}, "image": {"path"}, "callout": {"kind", "text"}, "table": {"header", "rows"},
    "demonstration": {"id"}, "check": {"id"}, "compare": {"demonstration", "changed", "cases", "takeaway"},
    "playground": {"exercise_id"},
    "walkthrough": {"exercise_id", "notes"}, "hero": {"headline"}, "gallery": {"items"},
}
RUNTIMES = {"pyodide", "display-only"}
PACKAGES_FILE = config.APP_DIR / "content" / "pyodide-packages.json"  # packages the setup script downloads
TIME_LIMIT_RANGE = (1000, 60000)
HELP_TYPES = {"smaller-example", "prerequisite", "geometry", "expanded-equation", "common-mistake", "code-connection"}
CHECK_KINDS = {"prediction", "new-case", "explain-back"}
ANSWER_TYPES = {"choice", "number", "shape", "text"}
VIEW_TYPES = {"grid", "tokens", "text", "equation", "plot", "graph", "code"}
PLOT_KINDS = {"line", "scatter", "step", "bar", "area", "polygon"}
COMPONENTS_FILE = config.APP_DIR / "content" / "components.json"  # names the frontend registry must provide
CALLOUT_KINDS = {"definition", "note", "mistake", "boundary", "example"}
SLUG = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
LINK = re.compile(r"\[\[(article|q):([^\]|]+)(?:\|[^\]]*)?\]\]")
TOP = {"id", "kind", "title", "revision", "review_state", "exercises", "prerequisites", "sources", "media", "blocks",
       "demonstrations", "checks", "playgrounds"}


class ArticleError(Exception):
    pass


def inline_texts(block: dict):
    t = block["type"]
    if t in ("paragraph", "heading", "callout"):
        yield block["text"]
        if t == "callout" and block.get("title"):
            yield block["title"]
    elif t == "list":
        yield from block["items"]
    elif t == "table":
        for row in [block["header"], *block["rows"]]:
            yield from row
    elif t == "compare":
        yield block["takeaway"]
    elif t == "walkthrough":
        for n in block["notes"]:
            yield n.get("text", "")
    if block.get("caption"):
        yield block["caption"]


def validate_article(a: dict, where: str) -> list[str]:
    errs = []
    e = lambda m: errs.append(f"{where}: {m}")
    for k in a.keys() - TOP:
        e(f"unknown field {k!r}")
    for k in ("id", "kind", "title", "revision", "review_state", "blocks"):
        if k not in a:
            e(f"missing {k!r}")
    if errs:
        return errs
    if not SLUG.match(a["id"]):
        e("id must be a lowercase slug")
    if a["kind"] not in ("lesson", "prerequisite"):
        e("kind must be lesson or prerequisite")
    if a["review_state"] not in ("draft", "reviewed", "complete"):
        e("bad review_state")
    if not isinstance(a["revision"], int) or a["revision"] < 1:
        e("revision must be a positive integer")
    media_paths = {m.get("path") for m in a.get("media", [])}
    for m in a.get("media", []):
        if not m.get("path") or not m.get("description") or not m.get("alt_text"):
            e(f"media {m.get('path')!r} needs path, description, and alt_text")
    for i, b in enumerate(a["blocks"]):
        t = b.get("type")
        if t not in BLOCK_FIELDS:
            e(f"block {i}: unknown type {t!r}")
            continue
        for k in b.keys() - BLOCK_FIELDS[t] - {"type"}:
            e(f"block {i} ({t}): unknown field {k!r}")
        for k in REQUIRED[t] - b.keys():
            e(f"block {i} ({t}): missing {k!r}")
        if t == "image" and b.get("path") and b["path"] not in media_paths:
            e(f"block {i}: image {b['path']!r} is not listed in media with alt text")
        if t == "gallery":
            if not (isinstance(b.get("items"), list) and b["items"]):
                e(f"block {i}: gallery needs items")
            for it in b.get("items", []):
                if it.get("path") not in media_paths or not it.get("caption"):
                    e(f"block {i}: gallery item {it.get('path')!r} needs a caption and an entry in media with alt text")
        if t == "callout" and b.get("kind") not in CALLOUT_KINDS:
            e(f"block {i}: callout kind must be one of {sorted(CALLOUT_KINDS)}")
        if t == "heading" and b.get("level") not in (2, 3):
            e(f"block {i}: heading level must be 2 or 3")
        if t == "table" and any(len(r) != len(b["header"]) for r in b["rows"]):
            e(f"block {i}: table row length differs from header")
    errs += validate_interactive(a, where)
    if a["kind"] == "lesson" and not any(x.get("purpose") == "deep-lesson" for x in a.get("exercises", [])):
        e("a lesson needs an exercises entry with purpose deep-lesson")
    for x in a.get("exercises", []):
        if not (isinstance(x.get("id"), int) and x.get("purpose") in ("deep-lesson", "prerequisite", "related")
                and isinstance(x.get("order"), int)):
            e(f"bad exercises entry {x}")
    return errs


def components() -> set[str]:
    return set(json.loads(COMPONENTS_FILE.read_text()))


def validate_concept_lab(did: str, d: dict, e) -> None:
    """Structure checks for a concept laboratory. Numbers are checked by the lesson build, not here."""
    data = d.get("data")
    if not isinstance(data, dict) or "cases" not in data:
        e(f"demonstration {did}: concept-lab component needs data.cases")
        return
    params = data.get("params", [])
    keys = [""]
    for p in params:
        opts = [o.get("value") for o in p.get("options", [])]
        if not p.get("name") or not p.get("label") or len(opts) < 2:
            e(f"demonstration {did}: each param needs name, label, and at least 2 options")
            return
        keys = [(k + "|" if k else "") + str(v) for k in keys for v in opts]
        if p["name"] not in d["input_schema"]["defaults"] or d["input_schema"]["defaults"][p["name"]] not in opts:
            e(f"demonstration {did}: default for param {p['name']!r} is missing or not an option")
    if sorted(data["cases"]) != sorted(keys):
        e(f"demonstration {did}: cases must cover exactly every parameter combination ({len(keys)} expected, got {len(data['cases'])})")
        return
    step_ids = {s["id"] for s in d.get("steps", [])}
    for key, case in data["cases"].items():
        frames = case.get("frames")
        if not frames:
            e(f"demonstration {did} case {key!r}: no frames")
            continue
        for f in frames:
            fw = f"demonstration {did} case {key!r} frame {f.get('id')!r}"
            for k in ("id", "title", "changed", "why", "fixed"):
                if not isinstance(f.get(k), str) or not f[k].strip():
                    e(f"{fw}: missing {k}")
            if f.get("id") not in step_ids:
                e(f"{fw}: frame id is not in the demonstration steps")
            ids, links = set(), []   # one id may label several cells or views: they are one object and select together
            views = f.get("views")
            if not views:
                e(f"{fw}: no views")
                continue
            for v in views:
                t = v.get("type")
                if t not in VIEW_TYPES:
                    e(f"{fw}: unknown view type {t!r}")
                    continue
                if t == "grid":
                    cells = v.get("cells")
                    if not cells or len({len(r) for r in cells}) != 1:
                        e(f"{fw}: grid rows must be non-empty and equally long")
                        continue
                    if sum(len(r) for r in cells) > 400:
                        e(f"{fw}: grid has more than 400 cells")
                    local = set()
                    for r in cells:
                        for c in r:
                            if isinstance(c, dict):
                                if "t" not in c:
                                    e(f"{fw}: grid cell needs t")
                                if c.get("id"):
                                    ids.add(c["id"])   # one id on several cells selects them together, for example a whole row
                                links += c.get("links", [])
                    for k in ("row_labels", "col_labels"):
                        if k in v and len(v[k]) != (len(cells) if k == "row_labels" else len(cells[0])):
                            e(f"{fw}: {k} length differs from the grid")
                elif t == "tokens":
                    if not v.get("items"):
                        e(f"{fw}: tokens view needs items")
                    local = set()
                    for c in v.get("items", []):
                        if c.get("id"):
                            ids.add(c["id"])
                        links += c.get("links", [])
                elif t == "plot":
                    for k in ("x", "y"):
                        r = v.get(k)
                        if not (isinstance(r, list) and len(r) == 2 and r[0] < r[1]):
                            e(f"{fw}: plot needs {k} range [min, max]")
                    for s in v.get("series", []):
                        if s.get("kind") not in PLOT_KINDS:
                            e(f"{fw}: plot series kind must be one of {sorted(PLOT_KINDS)}")
                        if not s.get("label"):
                            e(f"{fw}: every plot series needs a label")
                        for pt in s.get("points", []):
                            if len(pt) != 2 or not all(isinstance(q, (int, float)) and q == q and abs(q) < 1e12 for q in pt):
                                e(f"{fw}: plot points must be finite number pairs")
                                break
                    if not v.get("alt"):
                        e(f"{fw}: a plot needs alt text that states what it shows")
                elif t == "graph":
                    nodes = {n.get("id") for n in v.get("nodes", [])}
                    if not nodes or len(nodes) != len(v["nodes"]):
                        e(f"{fw}: graph nodes need unique ids")
                    for ed in v.get("edges", []):
                        if ed.get("from") not in nodes or ed.get("to") not in nodes:
                            e(f"{fw}: graph edge refers to an unknown node")
                    for n in v.get("nodes", []):
                        if n.get("sel"):
                            ids.add(n["sel"])
                        links += n.get("links", [])
                    if not v.get("alt"):
                        e(f"{fw}: a graph needs alt text that states what it shows")
                elif t in ("text", "equation"):
                    if not (v.get("text") or v.get("tex")):
                        e(f"{fw}: {t} view is empty")
                elif t == "code" and not v.get("code"):
                    e(f"{fw}: code view needs code")
            for l in links:
                if l not in ids:
                    e(f"{fw}: link to unknown id {l!r}")
            for v in views:
                for k in v.get("by_sel", {}):
                    if k not in ids:
                        e(f"{fw}: by_sel key {k!r} is not a selectable id")


SCENE_TYPES = {"cube", "text", "arrow", "line", "box", "plane", "dim"}
SCENE_PROPS = {"pos", "size", "scale", "opacity", "color", "edge", "label", "text", "visible", "ghost", "from", "to", "progress"}
SCENE_OBJECT_CAP = 220


def validate_scene_demo(did: str, d: dict, e) -> None:
    """Structure checks for the `scene` component: a timeline of chapters and keyframed 3D objects."""
    data = d.get("data")
    if not isinstance(data, dict) or "cases" not in data:
        e(f"demonstration {did}: scene component needs data.cases")
        return
    params = data.get("params", [])
    keys = [""]
    for p in params:
        opts = [o.get("value") for o in p.get("options", [])]
        if not p.get("name") or not p.get("label") or len(opts) < 2:
            e(f"demonstration {did}: each param needs name, label, and at least 2 options")
            return
        keys = [(k + "|" if k else "") + str(v) for k in keys for v in opts]
        if p["name"] not in d["input_schema"]["defaults"] or d["input_schema"]["defaults"][p["name"]] not in opts:
            e(f"demonstration {did}: default for param {p['name']!r} is missing or not an option")
    if sorted(data["cases"]) != sorted(keys):
        e(f"demonstration {did}: cases must cover exactly every parameter combination")
        return
    step_ids = {s["id"] for s in d.get("steps", [])}
    num = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) < 1e6
    vec3 = lambda v: isinstance(v, list) and len(v) == 3 and all(num(q) for q in v)
    for key, case in data["cases"].items():
        w = f"demonstration {did} case {key!r}"
        sc = case.get("scene")
        if not isinstance(sc, dict):
            e(f"{w}: missing scene")
            continue
        dur = sc.get("duration")
        if not (num(dur) and 2 <= dur <= 180):
            e(f"{w}: duration must be 2-180 seconds")
            continue
        if not sc.get("alt"):
            e(f"{w}: scene needs alt text that describes what the animation shows")
        chapters = sc.get("chapters") or []
        if not chapters:
            e(f"{w}: scene needs chapters")
        prev_end = 0
        for c in chapters:
            for k in ("id", "title", "changed", "why", "fixed"):
                if not isinstance(c.get(k), str) or not c[k].strip():
                    e(f"{w}: chapter {c.get('id')!r} is missing {k}")
            if c.get("id") not in step_ids:
                e(f"{w}: chapter id {c.get('id')!r} is not in the demonstration steps")
            if not (num(c.get("t0")) and num(c.get("t1")) and c["t0"] < c["t1"]):
                e(f"{w}: chapter {c.get('id')!r} needs t0 < t1")
                continue
            if abs(c["t0"] - prev_end) > 1e-6:
                e(f"{w}: chapter {c['id']!r} must start where the previous one ends ({prev_end})")
            prev_end = c["t1"]
        if chapters and abs(prev_end - dur) > 1e-6:
            e(f"{w}: the chapters must end at the scene duration ({dur})")
        objs = sc.get("objects") or []
        if not objs or len(objs) > SCENE_OBJECT_CAP:
            e(f"{w}: scene needs 1-{SCENE_OBJECT_CAP} objects")
        ids = [o.get("id") for o in objs]
        if len(set(ids)) != len(ids):
            e(f"{w}: duplicate object ids")
        for o in objs:
            ow = f"{w} object {o.get('id')!r}"
            if o.get("type") not in SCENE_TYPES:
                e(f"{ow}: type must be one of {sorted(SCENE_TYPES)}")
                continue
            for prop, ks in (o.get("tracks") or {}).items():
                if prop not in SCENE_PROPS:
                    e(f"{ow}: unknown track {prop!r}")
                    continue
                if not (isinstance(ks, list) and ks and all(isinstance(k, list) and len(k) >= 2 and num(k[0]) for k in ks)):
                    e(f"{ow}: track {prop!r} must be a list of [t, value] keys")
                    continue
                ts = [k[0] for k in ks]
                if ts != sorted(ts) or len(set(ts)) != len(ts):
                    e(f"{ow}: track {prop!r} keys must have strictly increasing times")
                if ts[0] < -1e-9 or ts[-1] > dur + 1e-6:
                    e(f"{ow}: track {prop!r} has keys outside 0..{dur}")
                for k in ks:
                    v = k[1]
                    if prop in ("pos", "from", "to", "size") and o["type"] != "box" and o["type"] != "plane" and not vec3(v) and prop != "size":
                        e(f"{ow}: track {prop!r} values must be [x, y, z]")
                        break
                    if prop in ("opacity", "progress", "scale") and not num(v):
                        e(f"{ow}: track {prop!r} values must be numbers")
                        break
            for k in ("pos", "from", "to"):
                if k in o and not vec3(o[k]):
                    e(f"{ow}: {k} must be [x, y, z]")
            for l in o.get("links", []):
                if l not in ids:
                    e(f"{ow}: link to unknown object {l!r}")
        for k in (sc.get("by_sel") or {}):
            if k not in ids:
                e(f"{w}: by_sel key {k!r} is not an object")
        if sc.get("style", "glass") not in ("glass", "concept"):
            e(f"{w}: style must be glass or concept")
        cam = sc.get("camera") or {}
        if cam.get("mode", "persp") not in ("persp", "ortho"):
            e(f"{w}: camera mode must be persp or ortho")
        for ck in cam.get("keys", []):
            if not (num(ck.get("t")) and vec3(ck.get("pos")) and vec3(ck.get("target"))) or ("zoom" in ck and not num(ck["zoom"])):
                e(f"{w}: camera keys need t, pos, and target (zoom is optional)")
                break
        for cid, cc in (sc.get("calc") or {}).items():
            if cid not in ids:
                e(f"{w}: calc key {cid!r} is not an object")
            if not (isinstance(cc, dict) and cc.get("terms") and len(cc.get("ops", [])) == len(cc["terms"]) - 1
                    and all(t.get("label") and "value" in t and t.get("tone") in ("a", "b", "c", "n") for t in cc["terms"])):
                e(f"{w}: calc {cid!r} needs terms (label, value, tone a|b|c|n) and one op between terms")
        if sc.get("calc_default") and sc["calc_default"] not in (sc.get("calc") or {}):
            e(f"{w}: calc_default is not in calc")
        for name, v in (sc.get("views") or {}).items():
            if not (vec3(v.get("pos")) and vec3(v.get("target"))):
                e(f"{w}: view {name!r} needs pos and target")


def validate_playgrounds(a: dict, where: str) -> list[str]:
    errs = []
    e = lambda m: errs.append(f"{where}: {m}")
    pgs = {p.get("exercise_id"): p for p in a.get("playgrounds", [])}
    if len(pgs) != len(a.get("playgrounds", [])):
        e("duplicate playground exercise_id")
    allowed = set(json.loads(PACKAGES_FILE.read_text()))
    for eid, p in pgs.items():
        for k in p.keys() - {"exercise_id", "runtime", "packages", "starter_code", "checks", "time_limit_ms", "max_input_size", "note"}:
            e(f"playground {eid}: unknown field {k!r}")
        if not isinstance(eid, int):
            e(f"playground exercise_id must be an integer: {eid!r}")
        elif eid not in {x.get("id") for x in a.get("exercises", [])}:
            e(f"playground {eid}: the article must list exercise {eid} in exercises")
        if p.get("runtime") not in RUNTIMES:
            e(f"playground {eid}: runtime must be one of {sorted(RUNTIMES)}")
        if p.get("runtime") == "pyodide":
            if not (p.get("starter_code") or "").strip():
                e(f"playground {eid}: a runnable playground needs starter_code")
            bad = set(p.get("packages", ["numpy"])) - allowed
            if bad:
                e(f"playground {eid}: packages not downloaded by the setup script: {sorted(bad)}")
        if p.get("runtime") == "display-only" and not p.get("note"):
            e(f"playground {eid}: a display-only playground needs a note that states why it does not run")
        t = p.get("time_limit_ms", 10000)
        if not (isinstance(t, int) and TIME_LIMIT_RANGE[0] <= t <= TIME_LIMIT_RANGE[1]):
            e(f"playground {eid}: time_limit_ms must be {TIME_LIMIT_RANGE[0]}-{TIME_LIMIT_RANGE[1]}")
        for c in p.get("checks", []):
            if not (c.get("name") and c.get("code")):
                e(f"playground {eid}: each check needs name and code")
    for i, b in enumerate(a.get("blocks", [])):
        if b.get("type") == "playground" and b.get("exercise_id") not in pgs:
            e(f"block {i}: playground for exercise {b.get('exercise_id')!r} is not defined in playgrounds")
    return errs


def validate_interactive(a: dict, where: str) -> list[str]:
    errs = validate_playgrounds(a, where)
    e = lambda m: errs.append(f"{where}: {m}")
    demos = {d.get("id"): d for d in a.get("demonstrations", [])}
    checks = {c.get("id"): c for c in a.get("checks", [])}
    if len(demos) != len(a.get("demonstrations", [])):
        e("duplicate demonstration id")
    if len(checks) != len(a.get("checks", [])):
        e("duplicate check id")
    known = components()
    for d in demos.values():
        did = d.get("id")
        if not isinstance(did, str) or not SLUG.match(did):
            e(f"demonstration id must be a slug: {did!r}")
        for k in d.keys() - {"id", "component", "input_schema", "seed", "steps", "expected", "selection_links", "help", "data"}:
            e(f"demonstration {did}: unknown field {k!r}")
        if d.get("component") not in known:
            e(f"demonstration {did}: unknown component {d.get('component')!r}")
        steps = d.get("steps")
        if d.get("component") == "concept-lab" and isinstance(d.get("input_schema"), dict) and "defaults" in d["input_schema"]:
            validate_concept_lab(did, d, e)
        if d.get("component") == "scene" and isinstance(d.get("input_schema"), dict) and "defaults" in d["input_schema"]:
            validate_scene_demo(did, d, e)
        if not isinstance(steps, list) or not steps or not all(s.get("id") and s.get("title") for s in steps):
            e(f"demonstration {did}: steps need id and title")
            steps = []
        step_ids = [s["id"] for s in steps]
        if len(set(step_ids)) != len(step_ids):
            e(f"demonstration {did}: duplicate step id")
        if not isinstance(d.get("input_schema"), dict) or "defaults" not in d.get("input_schema", {}):
            e(f"demonstration {did}: input_schema needs a defaults entry")
        for h in d.get("help", []):
            if h.get("step_id") not in step_ids:
                e(f"demonstration {did}: help refers to unknown step {h.get('step_id')!r}")
            if h.get("help_type") not in HELP_TYPES:
                e(f"demonstration {did}: help_type must be one of {sorted(HELP_TYPES)}")
            if not h.get("content"):
                e(f"demonstration {did}: help item has no content")
    for c in checks.values():
        cid = c.get("id")
        if c.get("kind") not in CHECK_KINDS:
            e(f"check {cid}: kind must be one of {sorted(CHECK_KINDS)}")
        if c.get("answer_type") not in ANSWER_TYPES:
            e(f"check {cid}: answer_type must be one of {sorted(ANSWER_TYPES)}")
        if not c.get("prompt"):
            e(f"check {cid}: missing prompt")
        ex = c.get("explanation") or {}
        if not ex.get("correct"):
            e(f"check {cid}: explanation.correct is required")
        if c.get("kind") != "explain-back":
            if "answer" not in c:
                e(f"check {cid}: missing answer")
            if not ex.get("default"):
                e(f"check {cid}: explanation.default is required for a checked answer")
            if c.get("answer_type") == "choice" and c.get("answer") not in c.get("options", []):
                e(f"check {cid}: answer must be one of the options")
        if c.get("demonstration") and c["demonstration"] not in demos:
            e(f"check {cid}: unknown demonstration {c['demonstration']!r}")
    for i, b in enumerate(a.get("blocks", [])):
        if b.get("type") == "demonstration" and b.get("id") not in demos:
            e(f"block {i}: unknown demonstration {b.get('id')!r}")
        if b.get("type") == "walkthrough":
            notes = b.get("notes")
            if not (isinstance(notes, list) and notes):
                e(f"block {i}: walkthrough needs notes")
                notes = []
            if b.get("exercise_id") not in {x.get("id") for x in a.get("exercises", [])}:
                e(f"block {i}: walkthrough exercise {b.get('exercise_id')!r} must be listed in exercises")
            for k, n in enumerate(notes):
                ln = n.get("lines")
                if not (isinstance(ln, list) and len(ln) == 2 and all(isinstance(v, int) for v in ln) and 1 <= ln[0] <= ln[1]):
                    e(f"block {i} note {k}: lines must be [first, last] with 1 <= first <= last")
                if not n.get("text") or not n.get("match"):
                    e(f"block {i} note {k}: needs text and match (code that must appear in those lines)")
        if b.get("type") == "check" and b.get("id") not in checks:
            e(f"block {i}: unknown check {b.get('id')!r}")
        if b.get("type") == "compare":
            if b.get("demonstration") not in demos:
                e(f"block {i}: compare refers to unknown demonstration {b.get('demonstration')!r}")
            if not (isinstance(b.get("cases"), list) and len(b["cases"]) == 2 and all("label" in c and "inputs" in c for c in b["cases"])):
                e(f"block {i}: compare needs exactly two cases with label and inputs")
    if a.get("review_state") == "complete" and a.get("kind") == "lesson":
        kinds = {c.get("kind") for c in checks.values()}
        if "prediction" not in kinds:
            e("a complete lesson needs a prediction check")
        if "new-case" not in kinds:
            e("a complete lesson needs a new-case check")
        if not any(b.get("type") == "walkthrough" for b in a.get("blocks", [])):
            e("a complete lesson needs a walkthrough that connects the supplied solution to the explanation")
        types = {(b.get("type"), b.get("kind")) for b in a.get("blocks", [])}
        if not ({("compare", None), ("callout", "mistake"), ("callout", "boundary")} & types):
            e("a complete lesson needs a comparison or a failure/boundary callout")
    return errs


def load_sources(folder: Path) -> list[dict]:
    items, errs = [], []
    for path in sorted(folder.glob("*.json")):
        try:
            a = json.loads(path.read_text())
        except json.JSONDecodeError as ex:
            errs.append(f"{path.name}: invalid JSON: {ex}")
            continue
        if path.stem != a.get("id"):
            errs.append(f"{path.name}: file name must equal the article id")
        errs += validate_article(a, path.name)
        items.append(a)
    ids = [a.get("id") for a in items]
    errs += [f"duplicate article id {i}" for i in {i for i in ids if ids.count(i) > 1}]
    if errs:
        raise ArticleError("\n".join(errs))
    return items


def cross_check(items: list[dict], existing_ids: set[str], existing_states: dict[str, str]) -> list[str]:
    errs = []
    by_id = {a["id"]: a for a in items}
    known = existing_ids | set(by_id)
    state = {**existing_states, **{a["id"]: a["review_state"] for a in items}}
    ex = sqlite3.connect(f"file:{config.EXERCISES_DB}?mode=ro", uri=True)
    exercise_ids = {r[0] for r in ex.execute("SELECT id FROM exercise")}
    demo_owner = {}
    for a in items:
        for d in a.get("demonstrations", []):
            if d.get("id") in demo_owner and demo_owner[d["id"]] != a["id"]:
                errs.append(f"demonstration id {d['id']!r} appears in {demo_owner[d['id']]!r} and {a['id']!r}")
            demo_owner.setdefault(d.get("id"), a["id"])
    for a in items:
        where = a["id"]
        for p in a.get("prerequisites", []):
            if p not in known:
                errs.append(f"{where}: unknown prerequisite {p!r}")
        for b in a["blocks"]:
            if b["type"] != "walkthrough" or b["exercise_id"] not in exercise_ids:
                continue
            code = ex.execute("SELECT solution_code FROM exercise WHERE id = ?", (b["exercise_id"],)).fetchone()[0].split("\n")
            for k, n in enumerate(b["notes"]):
                lo, hi = n["lines"]
                if hi > len(code):
                    errs.append(f"{where}: walkthrough note {k} ends at line {hi} but solution {b['exercise_id']} has {len(code)} lines")
                elif n["match"] not in "\n".join(code[lo - 1:hi]):
                    errs.append(f"{where}: walkthrough note {k}: {n['match']!r} is not in lines {lo}-{hi} of solution {b['exercise_id']}")
        for p in a.get("playgrounds", []):
            if p["exercise_id"] not in exercise_ids:
                errs.append(f"{where}: playground for unknown exercise {p['exercise_id']}")
        for x in a.get("exercises", []):
            if x["id"] not in exercise_ids:
                errs.append(f"{where}: unknown exercise {x['id']}")
        for d in a.get("demonstrations", []):
            for h in d.get("help", []):
                if h.get("prerequisite_id") and h["prerequisite_id"] not in known:
                    errs.append(f"{where}: help refers to unknown article {h['prerequisite_id']!r}")
                for kind, target in LINK.findall(h.get("content", "")):
                    if kind == "article" and target not in known:
                        errs.append(f"{where}: help links to unknown article {target!r}")
            if d["id"] in demo_owner and demo_owner[d["id"]] != a["id"]:
                errs.append(f"{where}: demonstration id {d['id']!r} is used by {demo_owner[d['id']]!r}")
        for b in a["blocks"]:
            for text in inline_texts(b):
                for kind, target in LINK.findall(text):
                    if kind == "article" and target not in known:
                        errs.append(f"{where}: link to unknown article {target!r}")
                    if kind == "q" and (not target.isdigit() or int(target) not in exercise_ids):
                        errs.append(f"{where}: link to unknown question {target!r}")
            if b["type"] == "image" and not (config.MEDIA_DIR / b["path"]).is_file():
                errs.append(f"{where}: media file missing: {b['path']}")
            if b["type"] == "gallery":
                for it in b["items"]:
                    if not (config.MEDIA_DIR / it["path"]).is_file():
                        errs.append(f"{where}: media file missing: {it['path']}")
        for m in a.get("media", []):
            p = m["path"]
            if p.startswith("/") or ".." in Path(p).parts:
                errs.append(f"{where}: media path must be relative: {p}")
        if a["review_state"] == "complete":
            for p in a.get("prerequisites", []):
                if state.get(p) != "complete":
                    errs.append(f"{where}: marked complete but prerequisite {p!r} is not complete")
    edges = {a["id"]: a.get("prerequisites", []) for a in items}
    if cycle := check_links.find_cycle(edges):
        errs.append("prerequisite cycle: " + " -> ".join(cycle))
    return errs


def source_hash(a: dict) -> str:
    doc = {k: v for k, v in a.items() if k != "revision"}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def apply(con: sqlite3.Connection, items: list[dict]) -> dict:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    result = {"added": [], "updated": [], "unchanged": []}
    for a in items:
        body = json.dumps(a["blocks"], ensure_ascii=False, sort_keys=True)
        sources = json.dumps(a.get("sources", []), ensure_ascii=False)
        row = con.execute("SELECT revision, source_hash FROM article WHERE id=?", (a["id"],)).fetchone()
        sh = source_hash(a)
        if row:
            same = row[1] == sh
            if a["revision"] < row[0]:
                raise ArticleError(f"{a['id']}: revision {a['revision']} is lower than stored revision {row[0]}")
            if a["revision"] == row[0] and not same:
                raise ArticleError(f"{a['id']}: content changed but revision is still {row[0]}; raise the revision")
            if a["revision"] == row[0]:
                result["unchanged"].append(a["id"])
                continue
            con.execute("UPDATE article SET kind=?, title=?, body=?, revision=?, review_state=?, sources=?, updated_at=?, "
                        "source_hash=? WHERE id=?",
                        (a["kind"], a["title"], body, a["revision"], a["review_state"], sources, now, sh, a["id"]))
            result["updated"].append(a["id"])
        else:
            con.execute("INSERT INTO article (id, kind, title, body, revision, review_state, sources, updated_at, source_hash) "
                        "VALUES (?,?,?,?,?,?,?,?,?)",
                        (a["id"], a["kind"], a["title"], body, a["revision"], a["review_state"], sources, now, sh))
            result["added"].append(a["id"])
    for a in items:
        if a["id"] in result["unchanged"]:
            continue
        con.execute("DELETE FROM exercise_article WHERE article_id=?", (a["id"],))
        con.execute("DELETE FROM article_prerequisite WHERE article_id=?", (a["id"],))
        con.execute("DELETE FROM article_media WHERE article_id=?", (a["id"],))
        con.execute("DELETE FROM step_help WHERE demonstration_id IN (SELECT id FROM demonstration WHERE article_id=?)", (a["id"],))
        con.execute("DELETE FROM demonstration WHERE article_id=?", (a["id"],))
        con.execute("DELETE FROM learning_check WHERE article_id=?", (a["id"],))
        for p in a.get("playgrounds", []):
            con.execute("INSERT OR REPLACE INTO playground (exercise_id, runtime, packages, starter_code, checks, "
                        "time_limit_ms, max_input_size, note) VALUES (?,?,?,?,?,?,?,?)",
                        (p["exercise_id"], p["runtime"], json.dumps(p.get("packages", ["numpy"])), p.get("starter_code"),
                         json.dumps(p.get("checks", [])), p.get("time_limit_ms", 10000), p.get("max_input_size"), p.get("note")))
        for d in a.get("demonstrations", []):
            con.execute("INSERT INTO demonstration (id, article_id, component, input_schema, seed, steps, expected, selection_links, data) "
                        "VALUES (?,?,?,?,?,?,?,?,?)",
                        (d["id"], a["id"], d["component"], json.dumps(d["input_schema"]), d.get("seed"),
                         json.dumps(d["steps"]), json.dumps(d.get("expected", [])), json.dumps(d.get("selection_links", [])),
                         json.dumps(d["data"]) if "data" in d else None))
            for h in d.get("help", []):
                con.execute("INSERT INTO step_help (demonstration_id, step_id, help_type, content, prerequisite_id) VALUES (?,?,?,?,?)",
                            (d["id"], h["step_id"], h["help_type"], h["content"], h.get("prerequisite_id")))
        for c in a.get("checks", []):
            con.execute("INSERT INTO learning_check VALUES (?,?,?,?,?,?)",
                        (a["id"], c["id"], c["kind"],
                         json.dumps({k: c[k] for k in ("prompt", "answer_type", "options", "demonstration", "case") if k in c}),
                         json.dumps({"value": c.get("answer"), "tolerance": c.get("tolerance", 0)}),
                         json.dumps(c["explanation"])))
        con.executemany("INSERT INTO exercise_article VALUES (?,?,?,?)",
                        [(x["id"], a["id"], x["purpose"], x["order"]) for x in a.get("exercises", [])])
        con.executemany("INSERT INTO article_prerequisite VALUES (?,?)",
                        [(a["id"], p) for p in a.get("prerequisites", [])])
        for m in a.get("media", []):
            con.execute("INSERT INTO media (path, description, alt_text, attribution) VALUES (?,?,?,?) "
                        "ON CONFLICT(path) DO UPDATE SET description=excluded.description, "
                        "alt_text=excluded.alt_text, attribution=excluded.attribution",
                        (m["path"], m["description"], m["alt_text"], m.get("attribution")))
            mid = con.execute("SELECT id FROM media WHERE path=?", (m["path"],)).fetchone()[0]
            con.execute("INSERT INTO article_media VALUES (?,?)", (a["id"], mid))
    return result


def run(check_only: bool = False) -> dict:
    folder = config.CONTENT_DIR / "articles"
    items = load_sources(folder) if folder.is_dir() else []
    db.init_content()
    con = db.open_content()
    try:
        existing = {r[0]: r[1] for r in con.execute("SELECT id, review_state FROM article")}
        errs = cross_check(items, set(existing), existing)
        if errs:
            raise ArticleError("\n".join(errs))
        if check_only:
            return {"checked": [a["id"] for a in items]}
        with con:  # one transaction
            result = apply(con, items)
        problems = check_links.check()
        if problems:
            raise ArticleError("link check failed after import:\n" + "\n".join(problems))
        return result
    finally:
        con.close()


if __name__ == "__main__":
    try:
        print(json.dumps(run(check_only=len(sys.argv) > 1 and sys.argv[1] == "check")))
    except ArticleError as ex:
        print(f"ARTICLE IMPORT FAILED\n{ex}", file=sys.stderr)
        sys.exit(1)
