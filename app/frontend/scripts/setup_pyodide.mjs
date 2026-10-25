// Copy the Pyodide runtime into public/pyodide and download the packages listed in app/content/pyodide-packages.json.
// This is the one required initial download. Run: npm run setup:pyodide
import { createHash } from "node:crypto";
import { cpSync, existsSync, mkdirSync, readFileSync, writeFileSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const src = join(root, "node_modules", "pyodide");
const out = join(root, "public", "pyodide");
const wanted = JSON.parse(readFileSync(join(root, "..", "content", "pyodide-packages.json")));
const VERSION = JSON.parse(readFileSync(join(src, "package.json"))).version;
const CDN = `https://cdn.jsdelivr.net/pyodide/v${VERSION}/full/`;

mkdirSync(out, { recursive: true });
for (const f of ["pyodide.js", "pyodide.asm.js", "pyodide.asm.wasm", "python_stdlib.zip", "pyodide-lock.json", "pyodide.mjs"]) {
  cpSync(join(src, f), join(out, f));
}
const lock = JSON.parse(readFileSync(join(out, "pyodide-lock.json")));

// Resolve dependencies from the lock file.
const need = new Set();
const visit = (n) => {
  const p = lock.packages[n];
  if (!p) throw new Error(`package ${n} is not in the Pyodide lock file`);
  if (need.has(n)) return;
  need.add(n);
  (p.depends || []).forEach(visit);
};
wanted.forEach(visit);

let total = 0;
for (const n of need) {
  const p = lock.packages[n];
  const dest = join(out, p.file_name);
  if (existsSync(dest) && createHash("sha256").update(readFileSync(dest)).digest("hex") === p.sha256) { total += statSync(dest).size; continue; }
  process.stdout.write(`downloading ${p.file_name} … `);
  const res = await fetch(CDN + p.file_name);
  if (!res.ok) throw new Error(`${res.status} for ${p.file_name}`);
  const buf = Buffer.from(await res.arrayBuffer());
  const hash = createHash("sha256").update(buf).digest("hex");
  if (hash !== p.sha256) throw new Error(`checksum mismatch for ${p.file_name}`);
  writeFileSync(dest, buf);
  total += buf.length;
  console.log(`${(buf.length / 1e6).toFixed(1)} MB, checksum ok`);
}
const core = ["pyodide.asm.wasm", "pyodide.asm.js", "python_stdlib.zip"].reduce((s, f) => s + statSync(join(out, f)).size, 0);
console.log(`Pyodide ${VERSION}: packages ${[...need].join(", ")}`);
console.log(`core ${(core / 1e6).toFixed(1)} MB + packages ${(total / 1e6).toFixed(1)} MB in ${out}`);
