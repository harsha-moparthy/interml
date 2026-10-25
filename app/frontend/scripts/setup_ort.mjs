// Copy the ONNX runtime files that the in-browser semantic search needs into public/ort. Run: npm run setup:ort
import { cpSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "node_modules", "onnxruntime-web", "dist");
const out = join(root, "public", "ort");
mkdirSync(out, { recursive: true });
for (const f of ["ort-wasm-simd-threaded.jsep.mjs", "ort-wasm-simd-threaded.jsep.wasm"]) cpSync(join(src, f), join(out, f));
console.log(`ONNX runtime copied to ${out}`);
