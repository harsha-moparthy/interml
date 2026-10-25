// In-browser query embedding for "search by meaning". Same model family as the backend (all-MiniLM-L6-v2, 8-bit quantised).
// The model and the ONNX runtime are served from this site (models/, ort/); nothing is fetched from another host.
import { asset } from "./staticApi.js";

export const WAIT_MS = 60000;   // a search typed while the model loads waits this long, then uses keywords only

// "idle" -> "loading" -> "ready" | "error". The index shows a note while it is "loading".
let status = "idle";
const listeners = new Set();
const setStatus = (s) => { status = s; listeners.forEach((l) => l()); };
export const meaningStatus = {
  get: () => status,
  subscribe: (l) => { listeners.add(l); return () => listeners.delete(l); },
};

let extractor;
export function loadEmbedder() {
  if (!extractor) {
    setStatus("loading");
    extractor = import("@huggingface/transformers").then(async ({ pipeline, env }) => {
      env.allowRemoteModels = false;
      env.allowLocalModels = true;
      env.localModelPath = asset("models/");
      env.backends.onnx.wasm.wasmPaths = asset("ort/");
      env.backends.onnx.wasm.numThreads = 1;   // threads need cross-origin isolation, which GitHub Pages cannot send
      const pipe = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { dtype: "q8", device: "wasm" });
      setStatus("ready");
      return pipe;
    }).catch((e) => { extractor = undefined; setStatus("error"); throw e; });
  }
  return extractor;
}

/** Start loading when the browser is idle, so the first search finds the model ready. */
export function preload() {
  if (status !== "idle") return;
  const go = () => loadEmbedder().catch(() => {});
  if (typeof requestIdleCallback === "function") requestIdleCallback(go, { timeout: 3000 });
  else setTimeout(go, 1500);
}

/** Unit-length vector for the query, or null when the model cannot load in time. */
export async function embedQuery(text, timeoutMs = WAIT_MS) {
  try {
    const ex = await Promise.race([loadEmbedder(), new Promise((_, no) => setTimeout(() => no(new Error("timeout")), timeoutMs))]);
    const out = await ex(text, { pooling: "mean", normalize: true });
    return Float32Array.from(out.data);
  } catch (e) {
    return null;
  }
}
