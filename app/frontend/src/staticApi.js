// Static mode (GitHub Pages): the same /api routes answered from JSON files in static-api/, and keyword search in the browser.
// Files come from `python -m placement.export_static`. Search by meaning needs the local model, so it is off here.
export const STATIC = import.meta.env?.VITE_STATIC === "1";
export const BASE = import.meta.env?.BASE_URL ?? "/";
/** Path of a bundled asset (media, runtime) that works under a sub-path such as /interml/. */
export const asset = (path) => `${BASE}${path.replace(/^\//, "")}`;

const files = new Map();
const load = (rel) => {
  if (!files.has(rel)) {
    files.set(rel, fetch(asset(`static-api/${rel}`)).then((r) => {
      if (!r.ok) throw Object.assign(new Error(`${r.status} ${r.statusText}`), { status: r.status, statusText: r.statusText });
      return r.json();
    }).catch((e) => { files.delete(rel); throw e; }));
  }
  return files.get(rel);
};

let vectors;
const embeddings = () => (vectors ||= fetch(asset("static-api/embeddings.bin")).then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(new Error(r.status)))).then((b) => new Float32Array(b)).catch(() => { vectors = undefined; return null; }));

/** Same contract as fetch() for the read-only /api routes. */
export async function apiFetch(url) {
  if (!STATIC) return fetch(url);
  try {
    const u = new URL(url, "http://local");
    const path = u.pathname.replace(/^\/api\//, "").replace(/\/$/, "");
    const body = path === "search" ? await search(u.searchParams)
      : path === "exercises" ? await load("all.json")
      : await load(`${path}.json`);
    return new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
  } catch (e) {
    return new Response("{}", { status: e.status || 404, statusText: e.statusText || "Not found" });
  }
}

// ------------------------------------------------------------------ search (port of backend/placement/search.py, keywords only)
import { embedQuery } from "./semantic.js";
const SEMANTIC_FLOOR = 0.25, SEMANTIC_MARGIN = 0.12, RRF_K = 60, DIM = 384;   // same values as the backend
const FILTER_KEYS = new Set(["chapter", "ch", "id", "lang", "source", "difficulty", "lesson", "has", "is"]);
const WEIGHT = { title: 10, chapter: 3, topics: 5, prompt: 2, code_terms: 1.5 };
const FIELDS = Object.keys(WEIGHT);

export function parse(q) {
  const out = { words: [], phrases: [], exclude: [], filters: {} };
  const re = /"([^"]+)"|(-?)(\w+):("[^"]+"|\S+)|(-)([\w-]+)|([\w'+#.\-/]+)/g;
  for (const m of q.matchAll(re)) {
    const [, phrase, , key, val, , negword, word] = m;
    if (phrase) out.phrases.push(phrase.toLowerCase());
    else if (key && FILTER_KEYS.has(key.toLowerCase())) (out.filters[key.toLowerCase()] ||= []).push(val.replace(/^"|"$/g, "").toLowerCase());
    else if (key) out.words.push(key.toLowerCase(), val.replace(/^"|"$/g, "").toLowerCase());
    else if (negword) out.exclude.push(negword.toLowerCase());
    else if (word) {
      const w = word.toLowerCase().replace(/^[.\-'/]+|[.\-'/]+$/g, "");
      if (w) out.words.push(w);
    }
  }
  return out;
}

const tokens = (text) => (text.toLowerCase().match(/[a-z0-9]+/g) || []);
const fts = (t) => t.replace(/[^\w]/g, " ").trim();

function leven(a, b) {
  const row = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    let prev = row[0];
    row[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const tmp = row[j];
      row[j] = Math.min(row[j] + 1, row[j - 1] + 1, prev + (a[i - 1] === b[j - 1] ? 0 : 1));
      prev = tmp;
    }
  }
  return row[b.length];
}

let indexPromise;
function buildIndex() {
  indexPromise ||= Promise.all([load("search-index.json"), load("all.json")]).then(([raw, all]) => {
    const lesson = new Map(all.items.map((e) => [e.id, e.lesson]));
    const order = all.items.map((e) => e.id);
    const df = new Map();
    const vocab = new Map();
    const docs = new Map();
    for (const d of raw.docs) {
      const toks = {};
      const seen = new Set();
      FIELDS.forEach((f, k) => {
        toks[f] = tokens(d[f] || "");
        for (const t of toks[f]) {
          seen.add(t);
          if (t.length >= 4 && /^[a-z]/.test(t)) vocab.set(t, (vocab.get(t) || 0) + [3, 2, 2, 1][k] || 1);
        }
      });
      for (const t of seen) df.set(t, (df.get(t) || 0) + 1);
      docs.set(d.id, { ...d, toks, lesson: lesson.get(d.id) || "none" });
    }
    return { ...raw, docs, order, df, vocab, n: raw.docs.length, stop: new Set(raw.stopwords), vocabWords: [...vocab.keys()] };
  });
  return indexPromise;
}

function correct(ix, words) {
  const fixes = {};
  for (const w of words) {
    if (w.length < 5 || ix.vocab.has(w) || ix.synonyms[w] || /^\d+$/.test(w)) continue;
    if (ix.vocabWords.some((v) => v.startsWith(w))) continue;   // still typing
    let best = null, bestScore = 0.75;
    for (const v of ix.vocabWords) {
      if (Math.abs(v.length - w.length) > 3) continue;
      const score = 1 - leven(w, v) / Math.max(w.length, v.length);
      if (score > bestScore || (score === bestScore && best && ix.vocab.get(v) > ix.vocab.get(best))) { best = v; bestScore = score; }
    }
    if (best) fixes[w] = best;
  }
  return fixes;
}

function expand(ix, words) {
  const present = new Set(words);
  const groups = words.map((w) => [w, ...(ix.synonyms[w] || "").split(" ").filter(Boolean)]);
  for (const [abbr, full] of Object.entries(ix.synonyms)) {
    const parts = full.split(" ").filter((p) => !ix.stop.has(p));
    if (!present.has(abbr) && parts.filter((p) => present.has(p)).length >= Math.min(2, parts.length)) groups.push([abbr]);
  }
  return groups;
}

const tokenMatches = (tok, term) => tok === term
  || (term.length >= 3 && tok.startsWith(term))
  || (tok.length >= 4 && term.startsWith(tok) && term.length - tok.length <= 3);
const countMatches = (list, term) => { let n = 0; for (const t of list) if (tokenMatches(t, term)) n++; return n; };

function allowedBy(ix, filters) {
  const tests = [];
  for (const [key, vals] of Object.entries(filters)) {
    for (const v of vals) {
      if (key === "chapter" || key === "ch") {
        const m = /^([ca]?)(\d{1,2})$/.exec(v);
        if (m) tests.push((d, c) => c.chapter_id === `${(m[1] || "c").toUpperCase()}${m[2].padStart(2, "0")}`);
        else tests.push((d, c) => c.chapter_title.toLowerCase().includes(v));
      } else if (key === "id") {
        const m = /^(\d+)(?:-(\d+))?$/.exec(v);
        if (m) tests.push((d) => d.id >= +m[1] && d.id <= +(m[2] || m[1]));
      } else if (key === "lang") {
        const names = { py: "Python", python: "Python", cpp: "C++", "c++": "C++", cuda: "C++" };
        tests.push((d) => d.language === (names[v] || v));
      } else if (key === "source") tests.push((d) => (d.source || "").toLowerCase() === v);
      else if (key === "difficulty") tests.push((d) => (d.difficulty || "").toLowerCase() === v);
      else if (key === "is" && (v === "repository" || v === "original")) tests.push((d) => d.provenance === v);
      else if (key === "lesson") tests.push((d) => (v === "any" ? d.lesson !== "none" : d.lesson === v));
      else if (key === "has" && v === "lesson") tests.push((d) => d.lesson !== "none");
    }
  }
  return tests;
}

const segments = (text, terms) => {
  const stems = [...new Set(terms.filter((t) => t.length >= 3).map((t) => t.slice(0, Math.max(3, t.length - 2))))].sort((a, b) => b.length - a.length);
  if (!stems.length) return [{ text, hit: false }];
  const re = new RegExp(`(${stems.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})\\w*`, "gi");
  const out = [];
  let pos = 0;
  for (const m of text.matchAll(re)) {
    if (m.index > pos) out.push({ text: text.slice(pos, m.index), hit: false });
    out.push({ text: m[0], hit: true });
    pos = m.index + m[0].length;
  }
  if (pos < text.length) out.push({ text: text.slice(pos), hit: false });
  return out;
};

const snippet = (prompt, terms, width = 170) => {
  const low = prompt.toLowerCase();
  const hits = terms.map((t) => low.indexOf(t.slice(0, Math.max(3, t.length - 2)))).filter((i) => i >= 0);
  const start = Math.max(0, (hits.length ? Math.min(...hits) : 0) - 40);
  return (start ? "…" : "") + prompt.slice(start, start + width).trim() + (start + width < prompt.length ? "…" : "");
};

/** Seams for tests: the query embedder can be replaced. */
export const hooks = { embed: embedQuery };

export async function search(params) {
  const ix = await buildIndex();
  const all = await load("all.json");
  const meta = new Map(all.items.map((e) => [e.id, e]));
  let q = params.get("q") || "";
  if (params.get("chapter")) q += ` chapter:${params.get("chapter")}`;
  if (params.get("lesson")) q += ` lesson:${params.get("lesson")}`;
  const limit = Math.min(+params.get("limit") || 40, all.items.length);
  const p = parse(q);
  const fixes = correct(ix, p.words);
  const words = p.words.map((w) => fixes[w] || w);
  const tests = allowedBy(ix, p.filters);
  const ok = (id) => tests.every((t) => t(ix.docs.get(id), meta.get(id)));
  const freeText = [...words, ...p.phrases].join(" ").trim();
  const note = { corrected: fixes, semantic: false, filters: p.filters, query: freeText, mode: params.get("mode") || "hybrid" };

  const mode = note.mode;
  let ids, total, why = {};
  if (!freeText && !p.exclude.length) {
    ids = ix.order.filter(ok);
    total = ids.length;
    ids = ids.slice(0, limit);
  } else {
    const groups = expand(ix, words).map((g) => [...new Set(g.flatMap((t) => fts(t).split(/\s+/)).filter((t) => t && !ix.stop.has(t)))]).filter((g) => g.length);
    const exclude = p.exclude.flatMap((x) => fts(x).split(/\s+/)).filter(Boolean);
    const excludedBy = (d) => exclude.some((x) => FIELDS.some((f) => countMatches(d.toks[f], x)));
    const kw = [];
    if (mode !== "semantic") {
      for (const id of ix.order) {
        const d = ix.docs.get(id);
        const hay = FIELDS.map((f) => d[f] || "").join(" \n ").toLowerCase();
        if (p.phrases.some((ph) => !hay.includes(ph)) || excludedBy(d)) continue;
        let score = 0;
        for (const g of groups) {
          for (const t of g) {
            const idf = Math.log(1 + ix.n / (1 + (ix.df.get(t) || 0)));
            for (const f of FIELDS) {
              const tf = countMatches(d.toks[f], t);
              if (tf) score += idf * WEIGHT[f] * ((tf * 2.2) / (tf + 1.2));
            }
          }
        }
        if (groups.length && score === 0) continue;
        if (!groups.length && !p.phrases.length) continue;
        kw.push([id, score]);
      }
      kw.sort((a, b) => b[1] - a[1]);
      kw.length = Math.min(kw.length, 60);
    }
    // Meaning: cosine similarity to the stored question vectors, as in backend/placement/search.py.
    let sem = [];
    if (mode !== "keyword" && freeText) {
      const [vec, mat] = await Promise.all([hooks.embed(freeText), embeddings()]);
      if (vec && mat) {
        const sims = ix.order.map((id, i) => {
          let dot = 0;
          for (let k = 0; k < DIM; k++) dot += mat[i * DIM + k] * vec[k];
          return [id, dot];
        }).sort((a, b) => b[1] - a[1]).slice(0, 40);
        const best = sims[0][1];
        sem = sims.filter(([, v]) => v >= (mode === "hybrid" ? SEMANTIC_FLOOR : 0) && (mode !== "hybrid" || v >= best - SEMANTIC_MARGIN));
        note.semantic = true;
      }
    }
    const score = new Map();
    const bump = (id, w, tag) => { score.set(id, (score.get(id) || 0) + w); (why[id] ||= []).push(tag); };
    kw.forEach(([id], rank) => bump(id, 1 / (RRF_K + rank), "keyword"));
    sem.forEach(([id], rank) => { if (!excludedBy(ix.docs.get(id))) bump(id, 1 / (RRF_K + rank), "meaning"); });
    for (const w of new Set([...p.words, ...words])) {   // a bare number is a question ID: pin it first
      if (/^\d+$/.test(w) && ix.docs.has(+w)) { score.set(+w, (score.get(+w) || 0) + 1); (why[+w] ||= []).unshift("id"); }
    }
    ids = [...score.entries()].sort((a, b) => b[1] - a[1]).map(([id]) => id).filter(ok);
    total = ids.length;
    ids = ids.slice(0, limit);
  }

  const stop = ix.stop;
  const terms = (note.query.toLowerCase().match(/\w+/g) || []).filter((w) => !stop.has(w));
  terms.push(...terms.flatMap((w) => (ix.synonyms[w] || "").split(" ").filter(Boolean)));
  const items = ids.map((id) => {
    const d = ix.docs.get(id), m = meta.get(id);
    return { ...m, title_parts: segments(m.title, terms), snippet_parts: segments(snippet(d.prompt || "", terms), terms), why: why[id] || [] };
  });
  return { ...note, query: q, total, items };
}
