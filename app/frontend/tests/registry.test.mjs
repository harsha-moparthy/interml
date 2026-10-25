import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { REGISTRY_SPECS } from "../src/demo/registry.js";

test("registry names equal app/content/components.json", () => {
  const names = JSON.parse(readFileSync(new URL("../../content/components.json", import.meta.url)));
  assert.deepEqual(Object.keys(REGISTRY_SPECS).sort(), [...names].sort());
});
