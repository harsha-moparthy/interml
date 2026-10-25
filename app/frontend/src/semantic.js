// In-browser query embedding for "search by meaning". Same model family as the backend (all-MiniLM-L6-v2, 8-bit quantised).
// The model and the ONNX runtime are served from this site (models/, ort/); nothing is fetched from another host.
import { asset } from "./staticApi.js";

let extractor;
export function loadEmbedder() {
  extractor ||= import("@huggingface/transformers").then(async ({ pipeline, env }) => {
    env.allowRemoteModels = false;
    env.allowLocalModels = true;
    env.localModelPath = asset("models/");
    env.backends.onnx.wasm.wasmPaths = asset("ort/");
    env.backends.onnx.wasm.numThreads = 1;   // threads need cross-origin isolation, which GitHub Pages cannot send
    return pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { dtype: "q8", device: "wasm" });
  }).catch((e) => { extractor = undefined; throw e; });
  return extractor;
}

/** Unit-length vector for the query, or null when the model cannot load. */
export async function embedQuery(text, timeoutMs = 20000) {
  try {
    const ex = await Promise.race([loadEmbedder(), new Promise((_, no) => setTimeout(() => no(new Error("timeout")), timeoutMs))]);
    const out = await ex(text, { pooling: "mean", normalize: true });
    return Float32Array.from(out.data);
  } catch (e) {
    return null;
  }
}
