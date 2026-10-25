import test from "node:test";
import assert from "node:assert/strict";
import { ease, sampleTrack, evalObject, evalScene, chapterIndex, relatedIds, fmtTime } from "../src/scene/timeline.js";

const close = (a, b, e = 1e-9) => assert.ok(Math.abs(a - b) < e, `${a} vs ${b}`);

test("easing starts at 0, ends at 1, and is symmetric for inout", () => {
  for (const n of ["linear", "in", "out", "inout"]) { close(ease(n, 0), 0); close(ease(n, 1), 1); }
  close(ease("inout", 0.5), 0.5);
  assert.ok(ease("in", 0.5) < 0.5 && ease("out", 0.5) > 0.5);
  close(ease("inout", -3), 0); close(ease("inout", 9), 1);
});

test("a number track holds before the first key, interpolates, and holds after the last", () => {
  const k = [[1, 0, "linear"], [3, 10, "linear"]];
  assert.equal(sampleTrack(k, 0, "opacity"), 0);
  close(sampleTrack(k, 2, "opacity"), 5);
  assert.equal(sampleTrack(k, 99, "opacity"), 10);
});

test("vectors and colours interpolate per component", () => {
  const v = sampleTrack([[0, [0, 0, 0], "linear"], [2, [2, 4, 6], "linear"]], 1, "pos");
  assert.deepEqual(v, [1, 2, 3]);
  assert.equal(sampleTrack([[0, "#000000", "linear"], [1, "#ffffff", "linear"]], 0.5, "color"), "#808080");
});

test("labels and visibility switch at the next key and never blend", () => {
  const k = [[0, "1"], [2, "2"], [4, "3"]];
  assert.equal(sampleTrack(k, 1.99, "label"), "1");
  assert.equal(sampleTrack(k, 2, "label"), "2");
  assert.equal(sampleTrack([[0, false], [5, true]], 4.9, "visible"), false);
});

test("the same time always gives the same state, in any order", () => {
  const o = { id: "a", type: "cube", pos: [0, 0, 0], opacity: 1, tracks: { pos: [[0, [0, 0, 0]], [4, [8, 0, 0]]], opacity: [[0, 0], [2, 1]] } };
  const a = evalObject(o, 1), b = (evalObject(o, 3), evalObject(o, 1));
  assert.deepEqual(a, b);
  assert.equal(evalObject(o, 4).pos[0], 8);
});

test("static fields are used when there is no track", () => {
  const s = evalObject({ id: "a", type: "cube", pos: [1, 2, 3], size: 0.9 }, 5);
  assert.deepEqual(s.pos, [1, 2, 3]); assert.equal(s.size, 0.9);
});

test("chapters are found by time, including the exact start", () => {
  const sc = { chapters: [{ t0: 0 }, { t0: 4 }, { t0: 9 }] };
  assert.equal(chapterIndex(sc, 0), 0); assert.equal(chapterIndex(sc, 3.99), 0);
  assert.equal(chapterIndex(sc, 4), 1); assert.equal(chapterIndex(sc, 100), 2);
});

test("camera keys interpolate", () => {
  const sc = { chapters: [{ t0: 0 }], objects: [], camera: { keys: [{ t: 0, pos: [0, 0, 10], target: [0, 0, 0], ease: "linear" }, { t: 2, pos: [10, 0, 10], target: [0, 0, 0], ease: "linear" }] } };
  assert.deepEqual(evalScene(sc, 1).camera.pos, [5, 0, 10]);
});

test("related ids follow links in both directions", () => {
  const sc = { objects: [{ id: "out", links: ["x", "y"] }, { id: "x" }, { id: "y" }, { id: "z", links: ["x"] }] };
  const r = relatedIds(sc, "x");
  assert.deepEqual([...r.linked].sort(), ["out", "z"]);
  assert.deepEqual([...relatedIds(sc, "out").linked].sort(), ["x", "y"]);
  assert.equal(relatedIds(sc, "nope").sel, null);
});

test("time formatting", () => { assert.equal(fmtTime(75.4), "1:15"); assert.equal(fmtTime(3), "0:03"); });
