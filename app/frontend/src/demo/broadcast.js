// Broadcasting demonstration. Values are fixed so every number can be checked by hand:
// A holds 1, 2, 3, ... in row-major order. B holds 10, 20, 30, ... in row-major order.
// All arrays are shown with 3 axes. Missing leading axes count as size 1.

export const MAX_DIMS = 3;
export const MAX_SIZE = 4;

export const pad = (shape) => [...Array(MAX_DIMS - shape.length).fill(1), ...shape];
const clampShape = (s) => {
  const arr = Array.isArray(s) ? s.slice(0, MAX_DIMS) : [1];
  const clean = arr.map((n) => Math.min(Math.max(Math.trunc(Number(n)) || 1, 1), MAX_SIZE));
  return clean.length ? clean : [1];
};
export const fmt = (shape) => (shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(", ")})`);

const valueOf = (kind, padded, idx) => {
  const [, d1, d2] = padded;
  const a = idx.map((v, d) => (padded[d] === 1 ? 0 : v));
  const flat = (a[0] * d1 + a[1]) * d2 + a[2];
  return kind === "A" ? flat + 1 : 10 * (flat + 1);
};
const srcIndex = (padded, idx) => idx.map((v, d) => (padded[d] === 1 ? 0 : v));

export function sanitize(inputs) {
  return { shapeA: clampShape(inputs.shapeA), shapeB: clampShape(inputs.shapeB) };
}

export function checkAxes(shapeA, shapeB) {
  const pa = pad(shapeA), pb = pad(shapeB);
  return pa.map((a, axis) => {
    const b = pb[axis];
    const status = a === b ? "equal" : a === 1 ? "stretch A" : b === 1 ? "stretch B" : "conflict";
    return { axis, fromRight: MAX_DIMS - axis, a, b, status, size: status === "conflict" ? null : Math.max(a, b) };
  });
}

export function resultShape(shapeA, shapeB) {
  const axes = checkAxes(shapeA, shapeB);
  return axes.some((x) => x.status === "conflict") ? null : axes.map((x) => x.size);
}

export function defaultSelection(inputs) {
  return resultShape(inputs.shapeA, inputs.shapeB) ? { i: 0, j: 0, k: 0 } : null;
}

export function normalizeSelection(inputs, sel) {
  const shape = resultShape(inputs.shapeA, inputs.shapeB);
  if (!shape) return null;
  const s = sel || { i: 0, j: 0, k: 0 };
  return { i: Math.min(s.i, shape[0] - 1), j: Math.min(s.j, shape[1] - 1), k: Math.min(s.k, shape[2] - 1) };
}

// 2D slice of an array (axes 1 and 2) at index `i` along axis 0 (clamped to 0 when axis 0 has size 1).
const slice2D = (kind, padded, i) => {
  const ii = padded[0] === 1 ? 0 : i;
  return Array.from({ length: padded[1] }, (_, j) => Array.from({ length: padded[2] }, (_, k) => valueOf(kind, padded, [ii, j, k])));
};

export function reference(inputs) {
  const { shapeA, shapeB } = inputs;
  const shape = resultShape(shapeA, shapeB);
  const conflicts = checkAxes(shapeA, shapeB).filter((x) => x.status === "conflict").map((x) => x.fromRight);
  if (!shape) return { valid: false, result_shape: null, conflict_from_right: conflicts, output: null };
  const pa = pad(shapeA), pb = pad(shapeB);
  const output = Array.from({ length: shape[0] }, (_, i) => Array.from({ length: shape[1] }, (_, j) =>
    Array.from({ length: shape[2] }, (_, k) => valueOf("A", pa, [i, j, k]) + valueOf("B", pb, [i, j, k]))));
  return { valid: true, result_shape: shape, conflict_from_right: [], output };
}

export const CODE = [
  "A = np.arange(1, 1 + A_size).reshape(shape_a)",
  "B = 10 * np.arange(1, 1 + B_size).reshape(shape_b)",
  "C = A + B",
  "C[i, j, k]  # = A[ia, ja, ka] + B[ib, jb, kb]",
];

export function compute(inputs, selection) {
  const { shapeA, shapeB } = inputs;
  const pa = pad(shapeA), pb = pad(shapeB);
  const axes = checkAxes(shapeA, shapeB);
  const shape = resultShape(shapeA, shapeB);
  const conflicts = axes.filter((x) => x.status === "conflict");
  const text = shape ? `${fmt(shapeA)} + ${fmt(shapeB)} → ${fmt(shape)}`
    : `${fmt(shapeA)} + ${fmt(shapeB)} → error: ${conflicts.map((c) => `axis ${c.fromRight} from the right has sizes ${c.a} and ${c.b}`).join("; ")}`;
  const frames = [
    {
      id: "align", title: "Write both shapes with 3 axes", codeLines: [0, 1],
      changed: `Shape A is written as ${fmt(pa)} and shape B as ${fmt(pb)}.`,
      why: "Broadcasting compares axes from the right. A missing axis counts as size 1.",
      fixed: "The sizes you entered and all values stay the same.",
      view: { kind: "align", shapeA, shapeB, padA: pa, padB: pb },
    },
    {
      id: "check-axes", title: "Compare each pair of axes", codeLines: [2],
      changed: shape ? "Each axis pair is equal or contains a 1, so the shapes are compatible."
        : `Axis ${conflicts[0].fromRight} from the right has sizes ${conflicts[0].a} and ${conflicts[0].b}. Neither size is 1.`,
      why: shape ? "A size of 1 stretches to match the other size. Equal sizes match as they are."
        : "Two sizes can pair only when they are equal or when one of them is 1.",
      fixed: "The shapes do not change in this step.",
      view: { kind: "axes", axes, ok: !!shape },
    },
  ];
  if (shape) {
    frames.push({
      id: "result-shape", title: "Read the result shape", codeLines: [2],
      changed: `The result shape is ${fmt(shape)}.`,
      why: "Each result axis takes the larger size of its pair.",
      fixed: "The input shapes stay the same.",
      view: { kind: "result", axes, shape },
    });
    const sel = selection || { i: 0, j: 0, k: 0 };
    const ia = srcIndex(pa, [sel.i, sel.j, sel.k]), ib = srcIndex(pb, [sel.i, sel.j, sel.k]);
    const a = valueOf("A", pa, [sel.i, sel.j, sel.k]), b = valueOf("B", pb, [sel.i, sel.j, sel.k]);
    frames.push({
      id: "trace-cell", title: "Trace one output cell", codeLines: [3],
      changed: `Output cell C[${sel.i}, ${sel.j}, ${sel.k}] = ${a} + ${b} = ${a + b}.`,
      why: `Axes of size 1 always read index 0. A is read at [${ia}] and B at [${ib}].`,
      fixed: "The shapes and all input values stay the same. Select another output cell to trace it.",
      view: {
        kind: "trace", shape, padA: pa, padB: pb, sel, ia, ib, a, b, out: a + b,
        sliceA: slice2D("A", pa, sel.i), sliceB: slice2D("B", pb, sel.i),
        sliceC: Array.from({ length: shape[1] }, (_, j) => Array.from({ length: shape[2] }, (_, k) =>
          valueOf("A", pa, [sel.i, j, k]) + valueOf("B", pb, [sel.i, j, k]))),
        iaSlice: pa[0] === 1 ? 0 : sel.i, ibSlice: pb[0] === 1 ? 0 : sel.i,
      },
    });
  }
  return { frames, summary: { valid: !!shape, text }, reference: reference(inputs) };
}

export const spec = {
  sanitize, defaultSelection, normalizeSelection,
  compute: (inputs, selection) => compute(inputs, selection),
};
