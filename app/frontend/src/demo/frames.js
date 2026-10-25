// Generic demonstration whose frames are precomputed by a lesson build script (with NumPy, pandas, or the supplied solution).
// data = { params: [{name, label, options: [{value, label}]}], cases: { "<value>|<value>": { frames: [...] } } }

export const paramKey = (params, inputs) => params.map((p) => String(inputs[p.name])).join("|");

export function selectableLinks(frame) {
  const nodes = new Map();   // id -> links
  // The same id in several views is one object. Their links are merged.
  const add = (id, links = []) => id && nodes.set(id, [...new Set([...(nodes.get(id) || []), ...links])]);
  for (const v of frame.views) {
    if (v.type === "grid") v.cells.flat().forEach((c) => typeof c === "object" && add(c.id, c.links));
    else if (v.type === "tokens") v.items.forEach((c) => add(c.id, c.links));
    else if (v.type === "graph") v.nodes.forEach((n) => add(n.sel, n.links));
  }
  return nodes;
}

// Ids to highlight for a selection: the id itself, what it links to, and what links to it.
export function highlightSet(frame, sel) {
  if (!sel) return { sel: null, linked: new Set() };
  const nodes = selectableLinks(frame);
  const linked = new Set();
  if (nodes.has(sel)) nodes.get(sel).forEach((l) => linked.add(l));
  for (const [id, links] of nodes) if (links.includes(sel)) linked.add(id);
  linked.delete(sel);
  return { sel: nodes.has(sel) ? sel : null, linked };
}

export function makeFramesSpec(demo) {
  const data = demo.data;
  const params = data.params || [];
  const defaults = demo.input_schema.defaults;
  const allIds = new Set();
  Object.values(data.cases).forEach((c) => c.frames.forEach((f) => selectableLinks(f).forEach((_, id) => allIds.add(id))));
  const snap = (p, v) => {
    const hit = p.options.find((o) => String(o.value) === String(v));
    return hit ? hit.value : defaults[p.name];
  };
  const sanitize = (inputs) => Object.fromEntries(params.map((p) => [p.name, snap(p, inputs[p.name])]));
  return {
    sanitize,
    defaultSelection: () => null,
    normalizeSelection: (_, sel) => (sel && allIds.has(sel) ? sel : null),
    compute(inputs) {
      const frames = data.cases[paramKey(params, sanitize(inputs))].frames;
      return { frames, summary: { valid: true, text: "" }, reference: null };
    },
  };
}
