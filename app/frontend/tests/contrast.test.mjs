// WCAG 2.2 AA contrast for every text/background token pair, in both themes, read straight from src/tokens.css.
// Also guards the two dark blocks (system preference and explicit choice) against drifting apart.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const css = readFileSync(new URL("../src/tokens.css", import.meta.url), "utf8");

function block(startMarker) {
  const at = css.indexOf(startMarker);
  assert.ok(at >= 0, `missing block: ${startMarker}`);
  const open = css.indexOf("{", at);
  let depth = 0, i = open;
  for (; i < css.length; i++) { if (css[i] === "{") depth++; if (css[i] === "}" && --depth === 0) break; }
  return css.slice(open + 1, i);
}
const decls = (text) => Object.fromEntries([...text.matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)].map((m) => [m[1], m[2].trim()]));

const shared = decls(block(":root {"));                                    // aliases and scales
const lightOnly = decls(block("/* ---- light ---- */\n:root {"));
const darkMedia = decls(block(":root:not([data-theme=\"light\"])"));
const darkExplicit = decls(block(":root[data-theme=\"dark\"]"));

export const themes = {
  light: { ...shared, ...lightOnly },
  dark: { ...shared, ...lightOnly, ...darkExplicit },   // dark overrides light; every key must be set again
};

function resolve(theme, value, seen = new Set()) {
  const m = value.match(/^var\((--[\w-]+)\)$/);
  if (!m) return value;
  assert.ok(!seen.has(m[1]), `alias loop at ${m[1]}`);
  seen.add(m[1]);
  assert.ok(theme[m[1]] !== undefined, `undefined token ${m[1]}`);
  return resolve(theme, theme[m[1]], seen);
}
const hex = (h) => {
  const s = h.replace("#", "");
  const f = s.length === 3 ? [...s].map((c) => c + c).join("") : s;
  return [0, 2, 4].map((i) => parseInt(f.slice(i, i + 2), 16) / 255);
};
const lum = ([r, g, b]) => {
  const c = (v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4);
  return 0.2126 * c(r) + 0.7152 * c(g) + 0.0722 * c(b);
};
export function ratio(fg, bg) {
  const [a, b] = [lum(hex(fg)), lum(hex(bg))];
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}

// [text token, background token, minimum ratio, where it is used]
const TEXT = 4.5, UI = 3;
const tok = ["keyword", "string", "number", "atom", "comment", "function", "def", "type", "property", "punct", "meta"].map((n) => [`--tok-${n}`, "--code-bg", TEXT, `code ${n}`]);
export const PAIRS = [
  ["--color-fg", "--color-bg", TEXT, "body text"],
  ["--color-fg", "--color-surface", TEXT, "text on cards"],
  ["--color-muted", "--color-bg", TEXT, "secondary text"],
  ["--color-muted", "--color-surface", TEXT, "secondary text on cards"],
  ["--color-muted", "--code-bg", TEXT, "reading meta on code-bg cards"],
  ["--color-heading", "--color-bg", TEXT, "headings"],
  ["--color-accent", "--color-bg", TEXT, "links"],
  ["--color-accent", "--color-surface", TEXT, "links on cards"],
  ["--color-accent", "--color-accent-bg", TEXT, "badges and chips"],
  ["--color-bg", "--color-accent", TEXT, "pressed buttons"],
  ["--color-err", "--color-bg", TEXT, "error text"],
  ["--color-ok", "--color-bg", TEXT, "success text"],
  ["--color-fg", "--color-warn-bg", TEXT, "notices"],
  ["--code-fg", "--code-bg", TEXT, "code text"],
  ["--code-gutter-fg", "--code-gutter-bg", TEXT, "line numbers"],
  ["--color-fg", "--hl-bg", TEXT, "highlighted lines"],
  ["--color-fg", "--hl-active", TEXT, "active highlighted line"],
  ["--teal", "--teal-bg", TEXT, "tone a"], ["--amber", "--amber-bg", TEXT, "tone b"], ["--violet", "--violet-bg", TEXT, "tone c"],
  ["--color-muted", "--panel", TEXT, "secondary text on panels"],
  ["--color-accent", "--color-bg", UI, "focus ring and active borders (non-text)"],
  ["--color-line", "--color-bg", 1.2, "hairline (decorative, informational floor)"],
  ...tok,
];

test("the print block repeats the light block exactly", () => {
  const print = decls(block("@media print { :root:root:root"));
  assert.deepEqual(print, lightOnly);
});

test("both dark blocks declare the same values", () => {
  assert.deepEqual(darkMedia, darkExplicit);
});

test("every light token that a dark theme must restyle is set in dark", () => {
  const themed = Object.keys(lightOnly).filter((k) => !/^--(fs|space|radius|dur|ease|font)/.test(k));
  const missing = themed.filter((k) => darkExplicit[k] === undefined && !["--image-bg"].includes(k));
  assert.deepEqual(missing, []);
});

for (const [name, theme] of Object.entries(themes)) {
  for (const [fg, bg, min, use] of PAIRS) {
    test(`${name}: ${fg} on ${bg} >= ${min} (${use})`, () => {
      const r = ratio(resolve(theme, theme[fg] ?? `var(${fg})`), resolve(theme, theme[bg] ?? `var(${bg})`));
      assert.ok(r >= min, `${r.toFixed(2)}:1 is below ${min}:1`);
    });
  }
}
