// Judge a learner's answer to a learning check. Pure functions.

export function parseShape(text) {
  const t = String(text).trim().toLowerCase();
  if (/^(error|none|incompatible|cannot|invalid)/.test(t)) return "error";
  const nums = t.match(/\d+/g);
  return nums ? `(${nums.map(Number).join(", ")})` : null;
}

export function normalize(answerType, value) {
  if (answerType === "shape") return parseShape(value);
  if (answerType === "number") {
    const n = Number(String(value).trim());
    return String(value).trim() === "" || Number.isNaN(n) ? null : n;
  }
  return String(value).trim();
}

// Returns { correct, message } or { empty: true }.
export function judge(check, guess) {
  const g = normalize(check.answer_type, guess);
  if (g === null || g === "") return { empty: true };
  const truth = normalize(check.answer_type, check.answer.value);
  const tol = check.answer.tolerance || 0;
  const correct = check.answer_type === "number" ? Math.abs(g - truth) <= tol : g === truth;
  const wrong = check.explanation.wrong || {};
  const message = correct ? check.explanation.correct : wrong[String(g)] ?? check.explanation.default;
  return { correct, message, normalized: g };
}
