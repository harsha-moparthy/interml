// Check each demonstration's `expected` values (from NumPy) against the browser computation.
// Usage: node tests/check_content.mjs <articles-dir>
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import assert from "node:assert/strict";
import { REGISTRY_SPECS } from "../src/demo/registry.js";
import { reference } from "../src/demo/broadcast.js";

const REFERENCES = { "broadcast-trace": reference };
const dir = process.argv[2];
if (!existsSync(dir)) { console.log("no articles yet"); process.exit(0); }
let checked = 0;
for (const f of readdirSync(dir).filter((x) => x.endsWith(".json"))) {
  const a = JSON.parse(readFileSync(join(dir, f)));
  for (const d of a.demonstrations || []) {
    assert.ok(d.component in REGISTRY_SPECS, `${f}: component ${d.component} is not registered`);
    // Data-driven components carry authored/precomputed frames. Their builders and
    // backend schema validate those values; they do not expose a browser reference
    // function like the broadcasting calculator does.
    if (REGISTRY_SPECS[d.component] === null) continue;
    for (const e of d.expected || []) {
      const spec = REGISTRY_SPECS[d.component];
      const inputs = spec.sanitize(e.inputs);
      assert.deepEqual(inputs, e.inputs, `${f}/${d.id}: inputs outside the allowed limits`);
      assert.deepEqual(REFERENCES[d.component](inputs), e.expect, `${f}/${d.id}: ${JSON.stringify(e.inputs)}`);
      checked++;
    }
  }
}
console.log(`expected values ok: ${checked} cases`);
