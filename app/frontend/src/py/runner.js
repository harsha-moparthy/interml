// Page-side controller for the Python worker. One shared runtime per page.
// States: idle -> loading -> ready -> running -> ready. stop() and timeouts end the worker and start a fresh one.

import { asset } from "../staticApi.js";

export const LIMITS = { defaultTimeoutMs: 10000, minTimeoutMs: 1000, maxTimeoutMs: 60000 };

export function createRunner({ workerUrl = asset("py-worker.js"), makeWorker = (u) => new Worker(u) } = {}) {
  let worker = null;
  let state = "idle";           // idle | loading | ready | running | unavailable
  let ready = null;             // promise for the current worker
  let pending = null;           // {id, resolve, timer}
  let seq = 0;
  let loadedPkgs = new Set();    // packages already loaded into the current worker
  const listeners = new Set();
  const notify = (extra = {}) => listeners.forEach((f) => f({ state, ...extra }));

  function start() {
    state = "loading"; notify();
    loadedPkgs = new Set();
    worker = makeWorker(workerUrl);
    ready = new Promise((resolve, reject) => {
      worker.onmessage = (e) => {
        const m = e.data;
        if (m.type === "ready") { state = "ready"; notify(); resolve(); }
        else if (m.type === "loaded" && pending?.loading) { pending.loading.forEach((p) => loadedPkgs.add(p)); pending.onLoaded(); }
        else if (m.type === "load-error" && pending?.loading) {
          const p = pending; pending = null; state = "ready"; notify();
          p.resolve({ status: "unavailable", message: `Could not load packages: ${m.message}` });
        }
        else if (m.type === "init-error") { state = "unavailable"; notify({ message: m.message }); reject(new Error(m.message)); }
        else if (m.type === "out" && pending && m.id === pending.id) pending.onOut?.(m);
        else if (m.type === "result" && pending && m.id === pending.id) {
          clearTimeout(pending.timer);
          const p = pending; pending = null;
          state = "ready"; notify();
          p.resolve({ status: "done", ...m });
        }
      };
      worker.onerror = (e) => {
        if (pending?.started) {                  // the worker died during a run, for example out of memory
          clearTimeout(pending.timer);
          const p = pending; pending = null;
          endWorker(); start();
          p.resolve({ status: "crashed", message: e.message || "The Python worker stopped unexpectedly." });
          return;
        }
        state = "unavailable"; notify({ message: e.message || "The Python worker failed to start." });
        reject(new Error(e.message || "worker error"));
      };
      worker.postMessage({ type: "init" });
    });
    ready.catch(() => {});
    return ready;
  }

  function endWorker() {
    worker?.terminate(); worker = null; ready = null;
  }

  // Resolves with {status: 'done'|'timeout'|'stopped'|'unavailable', ...}. The time limit counts only the run itself.
  function run(code, { checks = [], timeoutMs = LIMITS.defaultTimeoutMs, onOut, packages = [] } = {}) {
    if (pending) throw new Error("A run is already in progress.");
    const limit = Math.min(Math.max(timeoutMs, LIMITS.minTimeoutMs), LIMITS.maxTimeoutMs);
    if (state === "unavailable") endWorker();   // allow a retry after a failed load
    if (!worker) start();
    const id = ++seq;
    return new Promise((resolve) => {
      pending = { id, resolve, timer: null, onOut, loading: null, onLoaded: null };
      const begin = () => {
        if (pending?.id !== id) return;          // stopped while loading
        pending.loading = null;
        state = "running"; notify();
        pending.started = true;
        pending.timer = setTimeout(() => {
          pending = null;
          endWorker(); start();                  // a fresh runtime is ready for the next run
          resolve({ status: "timeout", limitMs: limit });
        }, limit);
        worker.postMessage({ type: "run", id, code, checks });
      };
      ready.then(() => {
        if (pending?.id !== id) return;          // stopped while the runtime was loading
        const need = packages.filter((p) => !loadedPkgs.has(p));
        if (!need.length) return begin();
        pending.loading = need;                  // package download is not counted in the time limit
        pending.onLoaded = begin;
        state = "loading"; notify();
        worker.postMessage({ type: "load", packages: need });
      }, (err) => {
        if (pending?.id !== id) return;
        pending = null;
        resolve({ status: "unavailable", message: err.message });
      });
    });
  }

  function stop() {
    if (!pending) return false;
    clearTimeout(pending.timer);
    const p = pending; pending = null;
    endWorker(); start();
    p.resolve({ status: "stopped" });
    return true;
  }

  return {
    run, stop,
    preload: () => { if (!worker) start(); return ready; },
    get state() { return state; },
    subscribe(f) { listeners.add(f); return () => listeners.delete(f); },
  };
}

let shared = null;
export const sharedRunner = () => (shared ??= createRunner());
