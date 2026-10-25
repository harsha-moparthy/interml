import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { initialState, makeReducer, framesOf } from "../src/demo/engine.js";
import { spec, compute, reference, checkAxes, resultShape } from "../src/demo/broadcast.js";
import { judge, parseShape } from "../src/demo/check.js";

const defaults = { shapeA: [2, 1, 3], shapeB: [1, 4, 1] };
const reducer = makeReducer(spec, defaults);
const run = (actions, s = initialState(spec, defaults)) => actions.reduce(reducer, s);
const expected = JSON.parse(readFileSync(new URL("./broadcast_expected.json", import.meta.url)));

test("initial state uses the documented defaults", () => {
  const s = initialState(spec, defaults);
  assert.deepEqual(s.inputs, defaults);
  assert.equal(s.step, 0);
  assert.deepEqual(s.selection, { i: 0, j: 0, k: 0 });
  assert.equal(s.playing, false);
});

test("forward then backward restores the earlier state exactly", () => {
  const s0 = initialState(spec, defaults);
  for (let n = 1; n <= 3; n++) {
    const forward = run(Array(n).fill({ type: "next" }), s0);
    const back = run(Array(n).fill({ type: "prev" }), forward);
    assert.deepEqual(back, s0);
    assert.deepEqual(framesOf(spec, back), framesOf(spec, s0));
  }
});

test("step never leaves the valid range", () => {
  const s = run(Array(20).fill({ type: "next" }));
  assert.equal(s.step, framesOf(spec, s).length - 1);
  assert.equal(run(Array(5).fill({ type: "prev" })).step, 0);
  assert.equal(run([{ type: "goto", step: 99 }]).step, 3);
});

test("reset restores inputs, step, selection, and help", () => {
  const edited = run([{ type: "setInput", name: "shapeB", value: [3, 4, 1] }, { type: "next" }, { type: "next" },
    { type: "select", selection: { i: 1, j: 2, k: 1 } }, { type: "toggleHelp" }]);
  assert.notDeepEqual(edited, initialState(spec, defaults));
  assert.deepEqual(run([{ type: "reset" }], edited), initialState(spec, defaults));
});

test("an edit that makes the shapes incompatible clamps the step and clears the selection", () => {
  const atEnd = run(Array(3).fill({ type: "next" }));
  assert.equal(atEnd.step, 3);
  const bad = run([{ type: "setInput", name: "shapeB", value: [4] }], atEnd);
  assert.equal(framesOf(spec, bad).length, 2);
  assert.equal(bad.step, 1);
  assert.equal(bad.selection, null);
  const fixed = run([{ type: "setInput", name: "shapeB", value: [3] }], bad);
  assert.deepEqual(fixed.selection, { i: 0, j: 0, k: 0 });
});

test("selection is clamped to the output shape", () => {
  const s = run([{ type: "select", selection: { i: 9, j: 9, k: 9 } }]);
  assert.deepEqual(s.selection, { i: 1, j: 3, k: 2 });
});

test("inputs are sanitized to the size limits", () => {
  const s = run([{ type: "setInput", name: "shapeA", value: [99, 0, -3, 7] }]);
  assert.deepEqual(s.inputs.shapeA, [4, 1, 1]);
});

test("play stops at the last step and pause stops it", () => {
  let s = run([{ type: "play" }]);
  assert.equal(s.playing, true);
  s = run(Array(3).fill({ type: "next" }), s);
  assert.equal(s.playing, false);
  assert.equal(run([{ type: "play" }], s).playing, false);
  assert.equal(run([{ type: "pause" }], run([{ type: "play" }])).playing, false);
});

test("help opening keeps the computation state", () => {
  const before = run([{ type: "next" }, { type: "select", selection: { i: 1, j: 1, k: 1 } }]);
  const after = run([{ type: "toggleHelp" }], before);
  assert.deepEqual({ ...after, helpOpen: false }, before);
});

test("every frame has text for what changed, why, and what stayed fixed", () => {
  for (const f of framesOf(spec, initialState(spec, defaults))) {
    for (const k of ["title", "changed", "why", "fixed"]) assert.ok(f[k] && f[k].length > 5, `${f.id}.${k}`);
  }
});

test("reference values match NumPy for 304 shape pairs", () => {
  assert.ok(expected.length >= 300);
  for (const c of expected) assert.deepEqual(reference(c.inputs), c.expect, JSON.stringify(c.inputs));
});

test("every traced cell matches NumPy, including the selected-cell sources", () => {
  for (const c of expected.filter((c) => c.expect.valid).slice(0, 60)) {
    const [I, J, K] = c.expect.result_shape;
    for (let i = 0; i < I; i++) for (let j = 0; j < J; j++) for (let k = 0; k < K; k++) {
      const t = compute(c.inputs, { i, j, k }).frames[3].view;
      assert.equal(t.out, c.expect.output[i][j][k]);
      assert.equal(t.sliceC[j][k], c.expect.output[i][j][k]);
      assert.equal(t.a + t.b, t.out);
    }
  }
});

test("the conflicting axis is named exactly", () => {
  const c = compute({ shapeA: [3], shapeB: [4] }).frames[1];
  assert.match(c.changed, /Axis 1 from the right has sizes 3 and 4/);
  assert.deepEqual(checkAxes([2, 3], [4, 3]).map((x) => x.status), ["equal", "conflict", "equal"]);
  assert.equal(resultShape([2, 3], [4, 3]), null);
});

test("worked example from the plan: (2,1,3)+(1,4,1) -> (2,4,3), cell (i,j,k) reads A[i,0,k] and B[0,j,0]", () => {
  assert.deepEqual(resultShape([2, 1, 3], [1, 4, 1]), [2, 4, 3]);
  const v = compute(defaults, { i: 1, j: 2, k: 1 }).frames[3].view;
  assert.deepEqual(v.ia, [1, 0, 1]);
  assert.deepEqual(v.ib, [0, 2, 0]);
  assert.equal(v.a, 5);   // A[1,0,1] = (1*1+0)*3+1 + 1 = 5
  assert.equal(v.b, 30);  // B[0,2,0] = 10*((0*4+2)*1+0+1) = 30
  assert.equal(v.out, 35);
});

const check = (over) => ({ answer_type: "shape", answer: { value: "(2, 4, 3)", tolerance: 0 },
  explanation: { correct: "Right.", default: "Not quite.", wrong: { "(2, 1, 3)": "That is shape A." } }, ...over });

test("shape answers accept varied formats and give specific feedback", () => {
  assert.equal(parseShape("2,4,3"), "(2, 4, 3)");
  assert.equal(parseShape("Cannot broadcast"), "error");
  assert.equal(judge(check(), "(2, 4, 3)").correct, true);
  assert.equal(judge(check(), "[2 4 3]").correct, true);
  assert.equal(judge(check(), "(2, 1, 3)").message, "That is shape A.");
  assert.equal(judge(check(), "(9, 9)").message, "Not quite.");
  assert.equal(judge(check(), "  ").empty, true);
});

test("number and choice answers", () => {
  const n = check({ answer_type: "number", answer: { value: 35, tolerance: 0.01 } });
  assert.equal(judge(n, "35.001").correct, true);
  assert.equal(judge(n, "36").correct, false);
  assert.equal(judge(n, "abc").empty, true);
  const c = check({ answer_type: "choice", answer: { value: "b" } });
  assert.equal(judge(c, "b").correct, true);
  assert.equal(judge(c, "a").correct, false);
});
