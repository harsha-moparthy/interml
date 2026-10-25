"""Hybrid question search: BM25 keyword ranking + local semantic vectors, fused with reciprocal rank fusion.

Query syntax (all optional, combine freely):
  chapter:attention  ch:C16        chapter by ID or words in its title
  id:316   id:300-350              question ID or range
  lang:python|cpp   source:dm|tt|resume|ext   difficulty:easy|medium|hard
  lesson:complete|planned|none     has:lesson   is:repository|original
  "exact phrase"    -excluded      other words are searched by meaning and by keyword
"""
import json
import re
import sqlite3
from collections import Counter
from difflib import get_close_matches
from functools import lru_cache

import numpy as np

from . import config, semantic

STOPWORDS = set("a an the of to in on for and or with by from is are be how do does what which that this it as at into "
                "i my me can should would using use make get when why vs via".split())

# Abbreviations a learner may type that titles spell out. The semantic model handles most meaning; this covers acronyms.
SYNONYMS = {
    "kl": "kullback leibler divergence", "bpe": "byte pair encoding", "svd": "singular value decomposition",
    "pca": "principal component analysis", "gmm": "gaussian mixture", "em": "expectation maximization",
    "lora": "low rank adaptation", "rope": "rotary positional embedding", "ddpm": "diffusion denoising",
    "vae": "variational autoencoder elbo", "dpo": "direct preference optimization", "ppo": "proximal policy optimization",
    "gae": "generalized advantage estimation", "ddp": "distributed data parallel all-reduce", "kv": "key value cache",
    "mha": "multi head attention", "mqa": "multi query attention", "gqa": "grouped query attention",
    "rnn": "recurrent", "lstm": "long short term memory", "gru": "gated recurrent unit", "cnn": "convolution",
    "nms": "non maximum suppression", "iou": "intersection over union", "bn": "batch normalization",
    "ln": "layer normalization", "sgd": "stochastic gradient descent", "lr": "learning rate", "mlp": "multilayer perceptron",
    "tfidf": "term frequency inverse document frequency", "auc": "area under roc curve", "roc": "receiver operating",
    "ece": "expected calibration error", "psi": "population stability index", "knn": "nearest neighbors",
    "svm": "support vector machine", "mle": "maximum likelihood", "map": "maximum a posteriori", "ab": "a/b test",
    "bm25": "ranking relevance", "ndcg": "ranking gain", "gpt": "decoder transformer", "vit": "vision transformer patch",
    "unet": "u-net segmentation", "dp": "differential privacy", "moe": "mixture of experts", "grad": "gradient",
    "backprop": "backpropagation gradient", "autograd": "automatic differentiation", "np": "numpy",
}

FILTER_KEYS = {"chapter", "ch", "id", "lang", "source", "difficulty", "lesson", "has", "is"}
W_BM25 = (0.0, 12.0, 10.0, 3.0, 5.0, 2.0, 1.5)   # id_text, title, chapter, topics, prompt, code_terms (first = unindexed id)


def _con() -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{config.EXERCISES_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


# ---------------------------------------------------------------- query parsing
def parse(q: str) -> dict:
    out = {"words": [], "phrases": [], "exclude": [], "filters": {}, "raw": q.strip()}
    for m in re.finditer(r'"([^"]+)"|(-?)(\w+):("[^"]+"|\S+)|(-)([\w-]+)|([\w\'+#.\-/]+)', q):
        phrase, neg, key, val, neg2, negword, word = m.groups()
        if phrase:
            out["phrases"].append(phrase.lower())
        elif key and key.lower() in FILTER_KEYS:
            out["filters"].setdefault(key.lower(), []).append(val.strip('"').lower())
        elif key:                                   # unknown key: treat as plain words
            out["words"] += [key.lower(), val.strip('"').lower()]
        elif negword:
            out["exclude"].append(negword.lower())
        elif word:
            w = word.lower().strip(".-'/")
            if w:
                out["words"].append(w)
    return out


def _fts_token(t: str) -> str:
    t = re.sub(r"[^\w]", " ", t).strip()
    return t


@lru_cache(maxsize=1)
def _vocabulary() -> Counter:
    """Original-spelling words from titles, topics, and prompts, used for spelling suggestions."""
    con = _con()
    c: Counter = Counter()
    for r in con.execute("SELECT title, chapter, topics, prompt FROM exercise_search"):
        for field, weight in zip(r, (3, 2, 2, 1)):
            for w in re.findall(r"[a-z][a-z0-9]{3,}", field.lower()):
                c[w] += weight
    return c


def correct_spelling(words: list[str]) -> dict[str, str]:
    """Map words that appear nowhere in the index to their closest indexed word."""
    vocab = _vocabulary()
    fixes = {}
    for w in words:
        if len(w) < 5 or w in vocab or w in SYNONYMS or w.isdigit():
            continue
        if any(v.startswith(w) for v in vocab):   # a prefix of a real word: the learner is still typing
            continue
        close = get_close_matches(w, vocab.keys(), n=3, cutoff=0.78)
        if close:
            fixes[w] = max(close, key=lambda c: vocab[c])
    return fixes


# ---------------------------------------------------------------- filters
def _allowed_ids(con, f: dict, lesson_status) -> set[int] | None:
    if not f:
        return None
    where, args = [], []
    for key, vals in f.items():
        for v in vals:
            if key in ("chapter", "ch"):
                m = re.fullmatch(r"([ca]?)(\d{1,2})", v)
                if m:   # bare numbers mean the main chapters (C); A01-A18 are the appendix chapters
                    where.append("e.chapter_id = ?"); args.append(f"{(m.group(1) or 'c').upper()}{int(m.group(2)):02d}")
                else:
                    where.append("lower(c.title) LIKE ?"); args.append(f"%{v}%")
            elif key == "id":
                m = re.fullmatch(r"(\d+)(?:-(\d+))?", v)
                if m:
                    lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
                    where.append("e.id BETWEEN ? AND ?"); args += [lo, hi]
            elif key == "lang":
                names = {"py": "Python", "python": "Python", "cpp": "C++", "c++": "C++", "cuda": "C++"}
                where.append("e.language = ?"); args.append(names.get(v, v))
            elif key == "source":
                where.append("lower(e.source_key) = ?"); args.append(v)
            elif key == "difficulty":
                where.append("lower(e.difficulty) = ?"); args.append(v)
            elif key == "is" and v in ("repository", "original"):
                where.append("e.provenance = ?"); args.append(v)
    sql = "SELECT e.id FROM exercise e JOIN chapter c ON c.id = e.chapter_id"
    if where:
        sql += " WHERE " + " AND ".join(where)
    ids = {r[0] for r in con.execute(sql, args)}
    for v in f.get("lesson", []) + (["any"] if "lesson" in f.get("has", []) else []):
        ids = {i for i in ids if (lesson_status(i) != "none" if v == "any" else lesson_status(i) == v)}
    return ids


# ---------------------------------------------------------------- keyword + semantic
def expand(words: list[str]) -> list[list[str]]:
    """For each query word: the word, its abbreviation's expansion, and the abbreviation when the words spell it out."""
    present = set(words)
    groups = [[w] + SYNONYMS.get(w, "").split() for w in words]
    for abbr, full in SYNONYMS.items():
        parts = [p for p in full.split() if p not in STOPWORDS]
        if abbr not in present and sum(p in present for p in parts) >= min(2, len(parts)):
            groups.append([abbr])
    return groups


def keyword_rank(con, words, phrases, exclude, limit=60) -> list[sqlite3.Row]:
    terms = []
    for expanded in expand(words):
        for t in expanded:
            t = _fts_token(t)
            if t and t not in STOPWORDS:
                terms.append(f'"{t}"*' if len(t) >= 3 else f'"{t}"')
    parts = []
    if terms:
        parts.append("(" + " OR ".join(dict.fromkeys(terms)) + ")")
    parts += [f'"{_fts_token(p)}"' for p in phrases if _fts_token(p)]
    if not parts:
        return []
    expr = " AND ".join(parts) if phrases else parts[0]
    ex_terms = [f'"{_fts_token(x)}"*' for x in exclude if _fts_token(x)]
    if ex_terms:
        expr += " NOT (" + " OR ".join(ex_terms) + ")"
    sql = ("SELECT exercise_id, bm25(exercise_search, ?, ?, ?, ?, ?, ?, ?) AS score FROM exercise_search "
           "WHERE exercise_search MATCH ? ORDER BY score LIMIT ?")
    try:
        return con.execute(sql, (*W_BM25, expr, limit)).fetchall()
    except sqlite3.OperationalError:
        return []


@lru_cache(maxsize=1)
def _matrix():
    con = _con()
    rows = con.execute("SELECT exercise_id, vector FROM exercise_embedding ORDER BY exercise_id").fetchall()
    if not rows:
        return None, None
    ids = np.array([r[0] for r in rows])
    mat = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    return ids, mat


def semantic_available() -> bool:
    ids, _ = _matrix()
    return ids is not None and semantic.available()


def semantic_rank(text: str, limit=40, floor=0.0, margin=None) -> list[tuple[int, float]]:
    ids, mat = _matrix()
    if ids is None or not semantic.available():
        return []
    sims = mat @ semantic.embed_query(text)
    order = np.argsort(-sims)[:limit]
    best = float(sims[order[0]])
    return [(int(ids[i]), float(sims[i])) for i in order
            if sims[i] >= floor and (margin is None or sims[i] >= best - margin)]


# ---------------------------------------------------------------- display helpers
def segments(text: str, terms: list[str]) -> list[dict]:
    """Split text into [{text, hit}] so the page can mark matches without inserting HTML."""
    stems = sorted({t[:max(3, len(t) - 2)] for t in terms if len(t) >= 3}, key=len, reverse=True)
    if not stems:
        return [{"text": text, "hit": False}]
    pat = re.compile("(" + "|".join(re.escape(s) for s in stems) + r")\w*", re.I)
    out, pos = [], 0
    for m in pat.finditer(text):
        if m.start() > pos:
            out.append({"text": text[pos:m.start()], "hit": False})
        out.append({"text": m.group(0), "hit": True})
        pos = m.end()
    if pos < len(text):
        out.append({"text": text[pos:], "hit": False})
    return out


def snippet(prompt: str, terms: list[str], width=170) -> str:
    low = prompt.lower()
    first = min((low.find(t[:max(3, len(t) - 2)]) for t in terms if low.find(t[:max(3, len(t) - 2)]) >= 0), default=0)
    start = max(0, first - 40)
    s = prompt[start:start + width].strip()
    return ("…" if start else "") + s + ("…" if start + width < len(prompt) else "")


# ---------------------------------------------------------------- main entry
SEMANTIC_FLOOR = 0.25     # absolute minimum cosine similarity
SEMANTIC_MARGIN = 0.12    # drop meaning-matches this far below the best one (chosen on tests/search_eval.json)
RRF_K = 60


def search(q: str, lesson_status=lambda i: "none", mode="hybrid", limit=50) -> dict:
    con = _con()
    p = parse(q)
    fixes = correct_spelling(p["words"]) if mode != "semantic" else {}
    words = [fixes.get(w, w) for w in p["words"]]
    allowed = _allowed_ids(con, p["filters"], lesson_status)
    free_text = " ".join(words + p["phrases"]).strip()
    note = {"corrected": fixes, "semantic": semantic_available(), "filters": p["filters"], "query": free_text}

    ids = [r[0] for r in con.execute("SELECT id FROM exercise ORDER BY list_position")]
    if not free_text and not p["exclude"]:
        pool = [i for i in ids if allowed is None or i in allowed]
        return {**note, "total": len(pool), "ids": pool[:limit], "why": {}, "scores": {}}

    kw = keyword_rank(con, words, p["phrases"], p["exclude"]) if mode in ("hybrid", "keyword") else []
    sem = semantic_rank(free_text, floor=SEMANTIC_FLOOR if mode == "hybrid" else 0.0,
                        margin=SEMANTIC_MARGIN if mode == "hybrid" else None) if (mode in ("hybrid", "semantic") and free_text) else []
    excluded = set()
    ex_terms = [f'"{_fts_token(x)}"*' for x in p["exclude"] if _fts_token(x)]
    if ex_terms and sem:   # keep semantic hits from re-adding excluded words
        excluded = {r[0] for r in con.execute("SELECT exercise_id FROM exercise_search WHERE exercise_search MATCH ?",
                                              (" OR ".join(ex_terms),))}
    score: dict[int, float] = {}
    why: dict[int, list[str]] = {}
    for rank, r in enumerate(kw):
        score[r[0]] = score.get(r[0], 0) + 1 / (RRF_K + rank)
        why.setdefault(r[0], []).append("keyword")
    for rank, (i, _) in enumerate(sem):
        if i in excluded:
            continue
        score[i] = score.get(i, 0) + 1 / (RRF_K + rank)
        why.setdefault(i, []).append("meaning")
    for w in dict.fromkeys(p["words"] + words):    # a bare number is a question ID: pin it first
        if w.isdigit() and int(w) in ids:
            score[int(w)] = score.get(int(w), 0) + 1.0
            why.setdefault(int(w), []).insert(0, "id")
    ranked = [i for i, _ in sorted(score.items(), key=lambda kv: -kv[1]) if allowed is None or i in allowed]
    return {**note, "total": len(ranked), "ids": ranked[:limit], "why": why,
            "scores": {i: round(score[i], 5) for i in ranked[:limit]}}


def present(result: dict, rows: dict[int, dict]) -> list[dict]:
    """Attach title/snippet segments to ranked IDs. `rows` maps id -> {title, prompt, ...}."""
    terms = [w for w in re.findall(r"\w+", result["query"].lower()) if w not in STOPWORDS]
    terms += [t for w in list(terms) for t in SYNONYMS.get(w, "").split()]
    items = []
    for i in result["ids"]:
        r = rows[i]
        d = {k: r[k] for k in ("id", "title", "chapter_id", "chapter_title", "subsection_title", "difficulty", "language")}
        d["title_parts"] = segments(r["title"], terms)
        d["snippet_parts"] = segments(snippet(r["prompt"], terms), terms)
        d["why"] = result["why"].get(i, [])
        items.append(d)
    return items
