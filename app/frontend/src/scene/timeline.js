// Keyframe timeline for 3D scenes. Pure functions, no rendering.
// A scene is { duration, chapters, objects, camera }. An object animates by `tracks`: { property: [[t, value, ease?], ...] }.
// A property without a track uses the object's static field of the same name.

export const EASE = {
  linear: (t) => t,
  in: (t) => t * t * t,
  out: (t) => 1 - (1 - t) ** 3,
  inout: (t) => (t < 0.5 ? 4 * t * t * t : 1 - ((-2 * t + 2) ** 3) / 2),
  step: (t) => (t < 1 ? 0 : 1),
};
export const ease = (name, t) => (EASE[name] || EASE.inout)(Math.min(Math.max(t, 0), 1));

const STEP_PROPS = new Set(["label", "text", "visible"]);
const hex = (c) => { const n = parseInt(c.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };
const toHex = (rgb) => "#" + rgb.map((v) => Math.round(v).toString(16).padStart(2, "0")).join("");

function mix(a, b, f) {
  if (typeof a === "number") return a + (b - a) * f;
  if (Array.isArray(a)) return a.map((v, i) => v + (b[i] - v) * f);
  if (typeof a === "string" && a[0] === "#") { const x = hex(a), y = hex(b); return toHex(x.map((v, i) => v + (y[i] - v) * f)); }
  return f < 1 ? a : b;
}

// Value of a track at time t. Before the first key it holds the first value. After the last key it holds the last value.
export function sampleTrack(keys, t, prop) {
  if (t <= keys[0][0]) return keys[0][1];
  const last = keys[keys.length - 1];
  if (t >= last[0]) return last[1];
  let i = 0;
  while (keys[i + 1][0] <= t) i++;
  const [t0, v0] = keys[i], [t1, v1, e] = keys[i + 1];
  if (STEP_PROPS.has(prop)) return v0;                 // text and visibility switch when the next key is reached
  return mix(v0, v1, ease(e || "inout", (t - t0) / (t1 - t0)));
}

export function evalObject(obj, t) {
  const out = {};
  for (const [k, v] of Object.entries(obj)) if (k !== "tracks") out[k] = v;
  for (const [prop, keys] of Object.entries(obj.tracks || {})) out[prop] = sampleTrack(keys, t, prop);
  return out;
}

export function chapterIndex(scene, t) {
  let idx = 0;
  scene.chapters.forEach((c, i) => { if (t >= c.t0 - 1e-9) idx = i; });
  return idx;
}

export function evalCamera(scene, t) {
  const keys = scene.camera?.keys;
  if (!keys?.length) return null;
  const pos = sampleTrack(keys.map((k) => [k.t, k.pos, k.ease]), t, "pos");
  const target = sampleTrack(keys.map((k) => [k.t, k.target, k.ease]), t, "target");
  const zoom = keys.some((k) => k.zoom) ? sampleTrack(keys.map((k) => [k.t, k.zoom ?? 1, k.ease]), t, "zoom") : null;
  return { pos, target, zoom };
}

export function evalScene(scene, t) {
  const objects = {};
  for (const o of scene.objects) objects[o.id] = evalObject(o, t);
  return { objects, camera: evalCamera(scene, t), chapter: chapterIndex(scene, t) };
}

// Ids to emphasise for a selection: the object, what it links to, and what links to it.
export function relatedIds(scene, sel) {
  if (!sel) return { sel: null, linked: new Set() };
  const byId = new Map(scene.objects.map((o) => [o.id, o]));
  if (!byId.has(sel)) return { sel: null, linked: new Set() };
  const linked = new Set(byId.get(sel).links || []);
  for (const o of scene.objects) if ((o.links || []).includes(sel)) linked.add(o.id);
  linked.delete(sel);
  return { sel, linked };
}

export const fmtTime = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
