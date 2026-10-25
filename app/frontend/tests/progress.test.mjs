import test, { mock } from "node:test";
import assert from "node:assert/strict";
import { createProgressStore, sanitize, STORAGE_KEY, MAX_RECENT } from "../src/progress.js";

function fakeStorage(initial = {}) {
  const map = new Map(Object.entries(initial));
  return {
    map,
    getItem: (k) => (map.has(k) ? map.get(k) : null),
    setItem: (k, v) => { map.set(k, String(v)); },
    removeItem: (k) => { map.delete(k); },
  };
}
let clock = 1000;
const tick = () => (clock += 1);
const make = (storage, opts) => createProgressStore(() => storage, { now: tick, ...opts });
const stored = (s) => JSON.parse(s.map.get(STORAGE_KEY));

test("fresh store is empty and persistent", () => {
  const p = make(fakeStorage());
  assert.deepEqual(p.getSnapshot(), { done: {}, bookmarks: {}, recent: [], reading: {}, persistent: true });
});

test("done toggles on and off and is written to storage", () => {
  const s = fakeStorage();
  const p = make(s);
  p.toggleDone(300);
  assert.ok(p.getSnapshot().done["300"] > 0);
  assert.deepEqual(Object.keys(stored(s).done), ["300"]);
  p.toggleDone(300);
  assert.deepEqual(p.getSnapshot().done, {});
  assert.deepEqual(stored(s).done, {});
});

test("a new store reads what an earlier one wrote", () => {
  const s = fakeStorage();
  const a = make(s);
  a.toggleDone(5); a.toggleBookmark(7); a.touch("question", 5); a.touch("article", "lesson-q016");
  const b = make(s);
  assert.deepEqual(Object.keys(b.getSnapshot().done), ["5"]);
  assert.deepEqual(Object.keys(b.getSnapshot().bookmarks), ["7"]);
  assert.deepEqual(b.getSnapshot().recent.map((r) => `${r.kind}:${r.id}`), ["article:lesson-q016", "question:5"]);
});

test("bookmarks toggle independently of done", () => {
  const p = make(fakeStorage());
  p.toggleBookmark(9);
  assert.deepEqual(Object.keys(p.getSnapshot().bookmarks), ["9"]);
  assert.deepEqual(p.getSnapshot().done, {});
  p.toggleBookmark(9);
  assert.deepEqual(p.getSnapshot().bookmarks, {});
});

test("recent keeps the latest ten, newest first, without duplicates", () => {
  const p = make(fakeStorage());
  for (let id = 1; id <= 12; id++) p.touch("question", id);
  const ids = p.getSnapshot().recent.map((r) => r.id);
  assert.equal(ids.length, MAX_RECENT);
  assert.deepEqual(ids, [12, 11, 10, 9, 8, 7, 6, 5, 4, 3]);
  p.touch("question", 5);
  assert.deepEqual(p.getSnapshot().recent.map((r) => r.id).slice(0, 2), [5, 12]);
  assert.equal(p.getSnapshot().recent.length, MAX_RECENT);
});

test("a question and an article with the same id text are different recent items", () => {
  const p = make(fakeStorage());
  p.touch("question", 5); p.touch("article", "a5");
  assert.equal(p.getSnapshot().recent.length, 2);
});

test("reading position is clamped and written after the debounce or on flush", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const s = fakeStorage();
    const p = make(s, { debounceMs: 400 });
    p.setReading("lesson-q016", 1.7);
    assert.equal(p.getSnapshot().reading["lesson-q016"].y, 1);
    assert.equal(s.map.has(STORAGE_KEY), false, "not written yet");
    mock.timers.tick(399);
    assert.equal(s.map.has(STORAGE_KEY), false);
    mock.timers.tick(1);
    assert.equal(stored(s).reading["lesson-q016"].y, 1);
    p.setReading("lesson-q016", -3);
    assert.equal(p.getSnapshot().reading["lesson-q016"].y, 0);
    p.flush();
    assert.equal(stored(s).reading["lesson-q016"].y, 0);
  } finally { mock.timers.reset(); }
});

test("reading ignores a bad article id or a non-number", () => {
  const p = make(fakeStorage());
  p.setReading("Bad Id!", 0.5); p.setReading("ok-id", NaN); p.setReading("ok-id", "0.5");
  assert.deepEqual(p.getSnapshot().reading, {});
});

test("subscribers hear every change and can unsubscribe", () => {
  const p = make(fakeStorage());
  let n = 0;
  const off = p.subscribe(() => n++);
  const before = p.getSnapshot();
  p.toggleDone(1);
  assert.equal(n, 1);
  assert.notEqual(p.getSnapshot(), before, "snapshot identity changes on change");
  off();
  p.toggleDone(2);
  assert.equal(n, 1);
});

test("reload picks up a write from another tab", () => {
  const s = fakeStorage();
  const p = make(s);
  s.map.set(STORAGE_KEY, JSON.stringify({ v: 1, done: { 42: 5 }, bookmarks: {}, recent: [], reading: {} }));
  p.reload();
  assert.deepEqual(Object.keys(p.getSnapshot().done), ["42"]);
});

// ---- bad stored data
test("corrupt JSON starts empty, stays persistent, and the next write replaces it", () => {
  const s = fakeStorage({ [STORAGE_KEY]: "{not json" });
  const p = make(s);
  assert.deepEqual(p.getSnapshot().done, {});
  assert.equal(p.getSnapshot().persistent, true);
  p.toggleDone(3);
  assert.deepEqual(Object.keys(stored(s).done), ["3"]);
});

test("a wrong version is ignored", () => {
  const p = make(fakeStorage({ [STORAGE_KEY]: JSON.stringify({ v: 2, done: { 1: 1 } }) }));
  assert.deepEqual(p.getSnapshot().done, {});
});

test("non-object records are ignored", () => {
  for (const raw of ["null", "42", '"text"', "[]", "true"]) {
    const p = make(fakeStorage({ [STORAGE_KEY]: raw }));
    assert.deepEqual(p.getSnapshot().done, {}, raw);
  }
});

test("wrong field types are dropped field by field", () => {
  const out = sanitize({
    v: 1,
    done: { 1: 10, abc: 5, 0: 5, "-3": 5, 7: "x", 8: Infinity, 9: -1, 1100: 2 },
    bookmarks: [1, 2],
    recent: [
      { kind: "question", id: 4, t: 9 },
      { kind: "question", id: "4", t: 9 },
      { kind: "page", id: 4, t: 9 },
      { kind: "article", id: "Bad Id", t: 9 },
      { kind: "article", id: "ok-id", t: "9" },
      { kind: "article", id: "ok-id", t: 8 },
      { kind: "article", id: "ok-id", t: 7 },
      null, 3,
    ],
    reading: { "ok-id": { y: 0.5, t: 3 }, "bad id": { y: 0.5, t: 3 }, over: { y: 2, t: 3 }, nan: { y: NaN, t: 3 }, missing: {} },
  });
  assert.deepEqual(out.done, { 1: 10, 1100: 2 });
  assert.deepEqual(out.bookmarks, {});
  assert.deepEqual(out.recent, [{ kind: "question", id: 4, t: 9 }, { kind: "article", id: "ok-id", t: 8 }]);
  assert.deepEqual(out.reading, { "ok-id": { y: 0.5, t: 3 } });
});

test("sanitize is idempotent and caps recent at ten", () => {
  const recent = Array.from({ length: 30 }, (_, i) => ({ kind: "question", id: i + 1, t: i }));
  const once = sanitize({ v: 1, done: { 3: 1 }, recent });
  assert.equal(once.recent.length, MAX_RECENT);
  assert.deepEqual(sanitize({ v: 1, ...once }), once);
});

// ---- storage failures
test("blocked storage: the getter throws, state lives in memory, persistent is false", () => {
  const p = createProgressStore(() => { throw new DOMException("blocked", "SecurityError"); }, { now: tick });
  assert.equal(p.getSnapshot().persistent, false);
  p.toggleDone(11); p.toggleBookmark(12); p.touch("question", 11);
  const snap = p.getSnapshot();
  assert.deepEqual(Object.keys(snap.done), ["11"]);
  assert.deepEqual(Object.keys(snap.bookmarks), ["12"]);
  assert.equal(snap.recent.length, 1);
  assert.equal(snap.persistent, false);
});

test("missing storage object behaves like blocked storage", () => {
  const p = createProgressStore(() => undefined, { now: tick });
  p.toggleDone(1);
  assert.equal(p.getSnapshot().persistent, false);
  assert.deepEqual(Object.keys(p.getSnapshot().done), ["1"]);
});

test("getItem that throws: starts empty and not persistent, still works", () => {
  const s = fakeStorage();
  s.getItem = () => { throw new Error("denied"); };
  const p = make(s);
  assert.equal(p.getSnapshot().persistent, false);
  p.toggleDone(2);
  assert.deepEqual(Object.keys(p.getSnapshot().done), ["2"]);
});

test("setItem that throws (quota): state kept, persistent false, recovers when a write succeeds", () => {
  const s = fakeStorage();
  let full = true;
  const real = s.setItem;
  s.setItem = (k, v) => { if (full) throw new DOMException("full", "QuotaExceededError"); real(k, v); };
  const p = make(s);
  p.toggleDone(1);
  assert.equal(p.getSnapshot().persistent, false);
  assert.deepEqual(Object.keys(p.getSnapshot().done), ["1"]);
  full = false;
  p.toggleDone(2);
  assert.equal(p.getSnapshot().persistent, true);
  assert.deepEqual(Object.keys(stored(s).done).sort(), ["1", "2"], "earlier change is written too");
});

// ---- export, import, reset
test("export, reset, import restores done, bookmarks, recent and reading", () => {
  const s = fakeStorage();
  const p = make(s);
  p.toggleDone(300); p.toggleDone(301); p.toggleBookmark(300); p.touch("article", "lesson-q016"); p.setReading("lesson-q016", 0.4); p.flush();
  const file = p.serialize();
  const before = { ...p.getSnapshot() };
  p.reset();
  assert.deepEqual(p.getSnapshot().done, {});
  assert.equal(s.map.has(STORAGE_KEY), false, "reset removes the stored record");
  const counts = p.importText(file);
  assert.deepEqual(counts, { done: 2, bookmarks: 1 });
  const after = p.getSnapshot();
  assert.deepEqual(after.done, before.done);
  assert.deepEqual(after.bookmarks, before.bookmarks);
  assert.deepEqual(after.recent, before.recent);
  assert.deepEqual(after.reading, before.reading);
  assert.deepEqual(Object.keys(stored(s).done).sort(), ["300", "301"], "import is written to storage");
});

test("import rejects bad files and leaves progress unchanged", () => {
  const p = make(fakeStorage());
  p.toggleDone(5);
  const before = p.getSnapshot();
  for (const bad of ["not json", "[]", "null", "42", JSON.stringify({ v: 2, done: { 1: 1 } }), JSON.stringify({ done: { 1: 1 } })]) {
    assert.throws(() => p.importText(bad), Error, bad);
  }
  assert.equal(p.getSnapshot(), before, "no change and no notification for rejected files");
});

test("import drops invalid entries instead of failing", () => {
  const p = make(fakeStorage());
  const counts = p.importText(JSON.stringify({ v: 1, done: { 1: 5, x: 5 }, bookmarks: "oops" }));
  assert.deepEqual(counts, { done: 1, bookmarks: 0 });
});

test("reset works when storage is blocked", () => {
  const p = createProgressStore(() => { throw new Error("blocked"); }, { now: tick });
  p.toggleDone(1);
  p.reset();
  assert.deepEqual(p.getSnapshot().done, {});
});

test("reset cancels a pending debounced write", () => {
  mock.timers.enable({ apis: ["setTimeout"] });
  try {
    const s = fakeStorage();
    const p = make(s);
    p.setReading("a", 0.5);
    p.reset();
    mock.timers.tick(1000);
    assert.equal(s.map.has(STORAGE_KEY), false);
  } finally { mock.timers.reset(); }
});
