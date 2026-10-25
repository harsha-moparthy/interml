// Learner progress, kept in this browser only: done and bookmarked questions, recent items, article reading position.
// The store never talks to the network. If storage is blocked, full or corrupt it keeps working in memory.

export const STORAGE_KEY = "placement.progress.v1";
export const MAX_RECENT = 10;

const emptyData = () => ({ done: {}, bookmarks: {}, recent: [], reading: {} });

const isQuestionKey = (k) => /^[1-9]\d{0,5}$/.test(k);
const isArticleId = (id) => typeof id === "string" && /^[a-z0-9-]{1,120}$/.test(id);
const isTime = (t) => typeof t === "number" && Number.isFinite(t) && t >= 0;

function cleanMarks(raw) {
  const out = {};
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    for (const [k, t] of Object.entries(raw)) if (isQuestionKey(k) && isTime(t)) out[k] = t;
  }
  return out;
}

function cleanRecent(raw) {
  if (!Array.isArray(raw)) return [];
  const seen = new Set();
  const items = [];
  for (const r of raw) {
    if (!r || typeof r !== "object" || !isTime(r.t)) continue;
    const ok = r.kind === "question" ? Number.isInteger(r.id) && isQuestionKey(String(r.id)) : r.kind === "article" && isArticleId(r.id);
    const key = `${r.kind}:${r.id}`;
    if (!ok || seen.has(key)) continue;
    seen.add(key);
    items.push({ kind: r.kind, id: r.id, t: r.t });
  }
  return items.sort((a, b) => b.t - a.t).slice(0, MAX_RECENT);
}

function cleanReading(raw) {
  const out = {};
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    for (const [id, r] of Object.entries(raw)) {
      if (isArticleId(id) && r && typeof r === "object" && Number.isFinite(r.y) && r.y >= 0 && r.y <= 1 && isTime(r.t)) out[id] = { y: r.y, t: r.t };
    }
  }
  return out;
}

/** Turn anything into valid progress data. Wrong version or shape gives empty data; bad fields are dropped one by one. */
export function sanitize(raw) {
  if (!raw || typeof raw !== "object" || raw.v !== 1) return emptyData();
  return { done: cleanMarks(raw.done), bookmarks: cleanMarks(raw.bookmarks), recent: cleanRecent(raw.recent), reading: cleanReading(raw.reading) };
}

/**
 * @param {() => Storage | null | undefined} getStorage  called on every access, because reading `localStorage` can throw
 * @param {{ now?: () => number, debounceMs?: number }} [options]
 */
export function createProgressStore(getStorage, { now = Date.now, debounceMs = 400 } = {}) {
  let data = emptyData();
  let persistent = true;
  let snapshot;
  let timer = null;
  const listeners = new Set();

  const makeSnapshot = () => { snapshot = { ...data, persistent }; };
  const emit = () => { makeSnapshot(); listeners.forEach((fn) => fn()); };

  function storage() {
    try { return getStorage() || null; } catch { return null; }
  }

  function load() {
    const s = storage();
    if (!s) { persistent = false; return emptyData(); }
    let text;
    try {
      text = s.getItem(STORAGE_KEY);
    } catch {
      persistent = false;   // reads fail, so assume writes will too
      return emptyData();
    }
    persistent = true;
    if (text == null) return emptyData();
    try {
      return sanitize(JSON.parse(text));
    } catch {
      return emptyData();   // corrupt record: storage works, so the next write replaces it
    }
  }

  function write() {
    if (timer) { clearTimeout(timer); timer = null; }
    const s = storage();
    if (!s) { persistent = false; return; }
    try {
      s.setItem(STORAGE_KEY, JSON.stringify({ v: 1, ...data }));
      persistent = true;
    } catch {
      persistent = false;
    }
  }

  function change(next, { debounce = false } = {}) {
    data = { ...data, ...next };
    if (debounce) {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => { write(); emit(); }, debounceMs);
    } else {
      write();
    }
    emit();
  }

  data = load();
  makeSnapshot();

  const toggle = (field, id) => {
    const key = String(id);
    const next = { ...data[field] };
    if (next[key] !== undefined) delete next[key]; else next[key] = now();
    change({ [field]: next });
  };

  return {
    getSnapshot: () => snapshot,
    subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); },
    toggleDone: (id) => toggle("done", id),
    toggleBookmark: (id) => toggle("bookmarks", id),
    /** Record that a question or article was opened. */
    touch(kind, id) {
      const rest = data.recent.filter((r) => !(r.kind === kind && r.id === id));
      change({ recent: [{ kind, id, t: now() }, ...rest].slice(0, MAX_RECENT) });
    },
    /** Save how far through an article the learner is (0..1). Writes are debounced. */
    setReading(articleId, y) {
      if (!isArticleId(articleId) || !Number.isFinite(y)) return;
      const clamped = Math.min(1, Math.max(0, y));
      change({ reading: { ...data.reading, [articleId]: { y: clamped, t: now() } } }, { debounce: true });
    },
    /** Write any pending debounced change now. */
    flush() { if (timer) { write(); emit(); } },
    /** Re-read storage, for example after the `storage` event from another tab. */
    reload() { data = load(); emit(); },
    /** The learner's progress as export-file text. */
    serialize: () => JSON.stringify({ v: 1, exportedAt: new Date(now()).toISOString(), ...data }, null, 2),
    /** Replace progress with the contents of an export file. Throws, and changes nothing, if the file is not one. */
    importText(text) {
      let raw;
      try { raw = JSON.parse(text); } catch { throw new Error("This file is not valid JSON."); }
      if (!raw || typeof raw !== "object" || Array.isArray(raw) || raw.v !== 1) throw new Error("This is not a progress file from this site.");
      change(sanitize(raw));
      return { done: Object.keys(data.done).length, bookmarks: Object.keys(data.bookmarks).length };
    },
    /** Delete all progress. The caller asks the learner to confirm first. */
    reset() {
      if (timer) { clearTimeout(timer); timer = null; }
      data = emptyData();
      const s = storage();
      try { s?.removeItem(STORAGE_KEY); } catch { persistent = false; }
      emit();
    },
  };
}

/** Save the export file through the browser. */
export function downloadExport(store, doc = document) {
  const url = URL.createObjectURL(new Blob([store.serialize()], { type: "application/json" }));
  const a = doc.createElement("a");
  a.href = url;
  a.download = `placement-progress-${new Date().toISOString().slice(0, 10)}.json`;
  doc.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// The store the app uses. Tabs stay in sync through the `storage` event.
export const progress = createProgressStore(() => globalThis.localStorage);

if (typeof window !== "undefined") {
  window.addEventListener("storage", (e) => { if (e.key === STORAGE_KEY || e.key === null) progress.reload(); });
  window.addEventListener("pagehide", () => progress.flush());
}
