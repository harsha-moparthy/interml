import test from "node:test";
import assert from "node:assert/strict";

// A tiny library of six questions, served through a fake fetch. Document i has a one-hot vector at position i,
// so a query vector that is one-hot at i is similar to document i only.
const DIM = 384;
const docs = [
  { id: 1, title: "KV-Cache Sizing", chapter: "LLMs, Transformers", topics: "memory inference", prompt: "Compute the key value cache size for a decoder.", code_terms: "", language: "Python", difficulty: "hard", source: "ext", provenance: "original" },
  { id: 2, title: "Batch Normalization", chapter: "Deep Learning", topics: "normalization training", prompt: "Implement batch normalization with running statistics.", code_terms: "", language: "Python", difficulty: "medium", source: "tt", provenance: "original" },
  { id: 3, title: "Principal Component Analysis", chapter: "Classical ML", topics: "dimensionality reduction", prompt: "Reduce dimensions with PCA via the SVD.", code_terms: "svd", language: "Python", difficulty: "medium", source: "tt", provenance: "original" },
  { id: 4, title: "Gradient Checkpointing", chapter: "GPU Systems", topics: "memory training", prompt: "Trade compute for memory during training.", code_terms: "", language: "C++", difficulty: "hard", source: "ext", provenance: "original" },
  { id: 5, title: "Softmax With Dropout", chapter: "Deep Learning", topics: "attention", prompt: "Apply softmax then dropout, excluding masked positions.", code_terms: "", language: "Python", difficulty: "easy", source: "dm", provenance: "repository" },
  { id: 6, title: "Causal Mask", chapter: "Appendix Practical", topics: "attention", prompt: "Build a causal mask for decoder attention.", code_terms: "", language: "Python", difficulty: "easy", source: "ext", provenance: "original" },
];
const chapterOf = { 1: ["A03", "LLMs"], 2: ["C05", "Deep Learning"], 3: ["C10", "Classical ML"], 4: ["A11", "GPU Systems"], 5: ["C05", "Deep Learning"], 6: ["A01", "Practical"] };
const all = {
  total: 6,
  items: docs.map((d) => ({ id: d.id, title: d.title, chapter_id: chapterOf[d.id][0], chapter_title: chapterOf[d.id][1], subsection_title: "s", difficulty: d.difficulty, language: d.language, lesson: d.id === 2 ? "complete" : "none" })),
};
const vectors = new Float32Array(6 * DIM);
docs.forEach((_, i) => { vectors[i * DIM + i] = 1; });
const files = {
  "/static-api/search-index.json": { synonyms: { kv: "key value cache", pca: "principal component analysis", bn: "batch normalization" }, stopwords: ["a", "the", "of", "to", "for", "use", "make"], docs },
  "/static-api/all.json": all,
};
globalThis.fetch = async (url) => {
  const path = new URL(url, "http://local").pathname;
  if (path === "/static-api/embeddings.bin") return new Response(vectors.buffer.slice(0));
  if (files[path]) return new Response(JSON.stringify(files[path]));
  return new Response("", { status: 404 });
};

const { search, parse, hooks } = await import("../src/staticApi.js");
const run = (q, extra = {}) => search(new URLSearchParams({ q, limit: "40", ...extra }));
const ids = (r) => r.items.map((i) => i.id);
const oneHot = (i) => { const v = new Float32Array(DIM); v[i] = 1; return v; };

test("parse reads words, phrases, exclusions and operators", () => {
  const p = parse('chapter:A03 "causal mask" -dropout softmax lang:cpp id:1-3');
  assert.deepEqual(p.filters, { chapter: ["a03"], lang: ["cpp"], id: ["1-3"] });
  assert.deepEqual(p.phrases, ["causal mask"]);
  assert.deepEqual(p.exclude, ["dropout"]);
  assert.deepEqual(p.words, ["softmax"]);
});

test("chapter operator accepts appendix and main chapters, and a bare number means C", async () => {
  hooks.embed = async () => null;
  assert.deepEqual(ids(await run("chapter:a03")), [1]);
  assert.deepEqual(ids(await run("ch:A01")), [6]);
  assert.deepEqual(ids(await run("chapter:5")).sort(), [2, 5]);
  assert.deepEqual(ids(await run("chapter:c05")).sort(), [2, 5]);
  assert.deepEqual(ids(await run("chapter:gpu")), [4]);
});

test("other operators filter", async () => {
  hooks.embed = async () => null;
  assert.deepEqual(ids(await run("lang:cpp")), [4]);
  assert.deepEqual(ids(await run("difficulty:easy")).sort(), [5, 6]);
  assert.deepEqual(ids(await run("id:2-3")), [2, 3]);
  assert.deepEqual(ids(await run("lesson:complete")), [2]);
  assert.deepEqual(ids(await run("is:repository")), [5]);
});

test("abbreviations expand and spelled-out words find the abbreviation", async () => {
  hooks.embed = async () => null;
  assert.equal(ids(await run("kv"))[0], 1);
  assert.equal(ids(await run("pca"))[0], 3);
  assert.equal(ids(await run("principal component"))[0], 3);
});

test("a misspelled word is corrected and reported", async () => {
  hooks.embed = async () => null;
  const r = await run("normalizaton");
  assert.deepEqual(r.corrected, { normalizaton: "normalization" });
  assert.equal(ids(r)[0], 2);
});

test("phrases and exclusions", async () => {
  hooks.embed = async () => null;
  assert.deepEqual(ids(await run('"causal mask"')), [6]);
  assert.ok(!ids(await run("softmax attention -dropout")).includes(5));
});

test("a bare number pins that question first", async () => {
  hooks.embed = async () => null;
  assert.equal(ids(await run("4 memory"))[0], 4);
  assert.ok((await run("4 memory")).items[0].why.includes("id"));
});

test("results carry highlighted title and snippet parts", async () => {
  hooks.embed = async () => null;
  const r = await run("cache");
  assert.ok(r.items[0].title_parts.some((p) => p.hit));
  assert.ok(r.items[0].snippet_parts.length > 0);
});

test("meaning search finds a question with no shared word and sets the flag", async () => {
  hooks.embed = async () => oneHot(3);   // pretend the query means question 4
  const r = await run("squeeze tensors smaller");
  assert.equal(r.semantic, true);
  assert.equal(ids(r)[0], 4);
  assert.deepEqual(r.items[0].why, ["meaning"]);
});

test("keyword and meaning hits are fused and both reasons are shown", async () => {
  hooks.embed = async () => oneHot(3);
  const r = await run("memory training");
  const top = r.items.find((i) => i.id === 4);
  assert.deepEqual(top.why.sort(), ["keyword", "meaning"]);
  assert.equal(ids(r)[0], 4);
});

test("meaning hits that match an excluded word are dropped", async () => {
  hooks.embed = async () => oneHot(3);
  assert.ok(!ids(await run("squeeze -checkpointing")).includes(4));
});

test("when the model is unavailable the list is keywords only", async () => {
  hooks.embed = async () => null;
  const r = await run("memory");
  assert.equal(r.semantic, false);
  assert.ok(ids(r).includes(4));
});

test("no query lists everything in order and the limit applies", async () => {
  hooks.embed = async () => null;
  const r = await search(new URLSearchParams({ q: "", limit: "3" }));
  assert.deepEqual([r.total, ids(r)], [6, [1, 2, 3]]);
});
