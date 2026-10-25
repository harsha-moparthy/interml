// Take the local service's top 10 for each query in parity_queries.json and store them in parity_reference.json.
// Needs the local service (default http://127.0.0.1:8765). Run: node parity_snapshot.mjs
// Run it again on purpose when the data or the ranking changes; the file is the reference that parity.mjs compares with.
import { readFileSync, writeFileSync } from "node:fs";
const API = process.env.API || "http://127.0.0.1:8765";
const queries = JSON.parse(readFileSync(new URL("./parity_queries.json", import.meta.url)));
const out = { api: API, taken: new Date().toISOString().slice(0, 10), results: {} };
for (const q of queries) {
  const r = await (await fetch(`${API}/api/search?${new URLSearchParams({ q, limit: 10 })}`)).json();
  out.results[q] = { total: r.total, top: r.items.map((i) => i.id), semantic: r.semantic };
}
const off = Object.entries(out.results).filter(([, v]) => !v.semantic).length;
if (off) { console.error(`${off} queries ran without search by meaning; start the service with the model installed`); process.exit(1); }
writeFileSync(new URL("./parity_reference.json", import.meta.url), JSON.stringify(out, null, 1) + "\n");
console.log(`wrote ${queries.length} queries`);
