export const A = [[1, 2, 3], [4, 5, 6]];
export const B = [10, 20, 30, 40];
export function outputValue(i, j, k, offset = 0) { return A[i][k] + B[j] + offset; }
export function broadcastShape(a, b) {
  const n = Math.max(a.length, b.length), out = [];
  for (let axis = 0; axis < n; axis++) {
    const x = a[a.length - n + axis] ?? 1, y = b[b.length - n + axis] ?? 1;
    if (x !== y && x !== 1 && y !== 1) return { error: `Aligned axis ${axis}: ${x} and ${y} differ, and neither is 1.` };
    out.push(x === 1 ? y : x);
  }
  return { shape: out };
}
export function parseShape(s) {
  if (!/^\s*\d+(\s*,\s*\d+)*\s*$/.test(s)) return null;
  const a = s.split(',').map(Number);
  return a.length <= 5 && a.every(n => Number.isSafeInteger(n) && n >= 0 && n <= 1000) ? a : null;
}
