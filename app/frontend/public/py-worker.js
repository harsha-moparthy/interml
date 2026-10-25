// Web Worker that runs learner Python with Pyodide. Classic worker so it loads from /public without bundling.
// Messages in:  {type:'init'}  {type:'load', packages}  {type:'run', id, code, checks}
// Messages out: {type:'ready'} {type:'init-error', message} {type:'loaded'} {type:'load-error', message}
//               {type:'out', id, stream, text}
//               {type:'result', id, ok, error, checks, ms}
/* global loadPyodide */
const post = self.postMessage.bind(self);
const MAX_OUTPUT = 20000;   // characters sent to the page per run
const MAX_CODE = 20000;     // characters of learner code
let pyodide = null;
let runId = null, emitted = 0, truncated = false;

function emit(stream, text) {
  if (runId === null || truncated) return;
  if (emitted + text.length > MAX_OUTPUT) {
    text = text.slice(0, Math.max(0, MAX_OUTPUT - emitted)) + `\n[output cut at ${MAX_OUTPUT} characters]`;
    truncated = true;
  }
  emitted += text.length;
  post({ type: "out", id: runId, stream, text: text + (truncated ? "" : "\n") });
}

// Keep the traceback from the learner's own code. Drop Pyodide's internal frames.
function tidy(message) {
  const lines = String(message).split("\n");
  const i = lines.findIndex((l) => l.includes('File "<exec>"') || l.includes('File "<learner>"'));
  const kept = i >= 0 ? ["Traceback (most recent call last):", ...lines.slice(i)] : lines;
  return kept.join("\n").trim();
}

const LOCKED = ["fetch", "XMLHttpRequest", "WebSocket", "EventSource", "importScripts", "indexedDB", "caches",
                "BroadcastChannel", "SharedWorker", "Worker"];
const originals = {};

// Precaution: learner code gets no network or worker APIs. This is not a security boundary.
// The properties stay writable so that package loading can restore them for a moment.
function lock() {
  for (const name of LOCKED) {
    if (!(name in originals)) originals[name] = self[name];
    try { Object.defineProperty(self, name, { value: undefined, writable: true, configurable: false }); } catch (_) { /* already defined */ }
  }
}
function unlock() {
  for (const name of LOCKED) { try { self[name] = originals[name]; } catch (_) { /* ignore */ } }
}

async function init() {
  try {
    importScripts("pyodide/pyodide.js");   // relative to this file, so a sub-path deployment works
    pyodide = await loadPyodide({ indexURL: new URL("pyodide/", self.location.href).href, stdout: (s) => emit("stdout", s), stderr: (s) => emit("stderr", s) });
    lock();
    post({ type: "ready" });
  } catch (e) {
    post({ type: "init-error", message: String(e && e.message || e) });
  }
}

// Packages download here, before the timed run, with network access restored only for this call.
async function load(packages) {
  unlock();
  try {
    await pyodide.loadPackage(packages);
    // Import the packages now. Importing pandas takes seconds in WebAssembly and is not the learner's computation,
    // so it must not count against the time limit. The modules stay loaded between runs; learner variables do not.
    for (const name of packages) {
      try { await pyodide.runPythonAsync(`import ${name.replace(/-/g, "_")}`); } catch (_) { /* dependency-only package */ }
    }
    post({ type: "loaded" });
  } catch (e) {
    post({ type: "load-error", message: String(e && e.message || e) });
  } finally {
    lock();
  }
}

async function run({ id, code, checks }) {
  const started = performance.now();
  runId = id; emitted = 0; truncated = false;
  const result = { type: "result", id, ok: true, error: null, checks: [] };
  if (code.length > MAX_CODE) {
    runId = null;
    post({ ...result, ok: false, error: `The code has ${code.length} characters. The limit is ${MAX_CODE}.`, ms: 0 });
    return;
  }
  const ns = pyodide.globals.get("dict")();   // fresh namespace for every run
  try {
    await pyodide.runPythonAsync(code, { globals: ns });
    for (const c of checks || []) {
      try {
        await pyodide.runPythonAsync(c.code, { globals: ns });
        result.checks.push({ name: c.name, ok: true });
      } catch (e) {
        const last = String(e.message).trim().split("\n").pop();
        result.checks.push({ name: c.name, ok: false, error: last });
      }
    }
  } catch (e) {
    result.ok = false;
    result.error = tidy(e.message);
  } finally {
    ns.destroy();
    runId = null;
  }
  post({ ...result, ms: Math.round(performance.now() - started) });
}

self.onmessage = (e) => {
  if (e.data.type === "init") init();
  else if (e.data.type === "load") load(e.data.packages);
  else if (e.data.type === "run") run(e.data);
};
