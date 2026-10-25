// three.js renderer for timeline scenes. Loaded on demand so that pages without a 3D scene never download three.js.
import { evalScene, relatedIds } from "./timeline.js";

const CAP = 220;   // objects per scene (checked by the import too)

export async function createSceneRenderer(host, { onPick, onHover } = {}) {
  const THREE = await import("three");
  const { OrbitControls } = await import("three/examples/jsm/controls/OrbitControls.js");
  const { RoundedBoxGeometry } = await import("three/examples/jsm/geometries/RoundedBoxGeometry.js");

  const canvas = document.createElement("canvas");
  canvas.className = "scene-canvas";
  host.appendChild(canvas);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  const scene3 = new THREE.Scene();
  const persp = new THREE.PerspectiveCamera(36, 1, 0.1, 300);
  const ortho = new THREE.OrthographicCamera(-8, 8, 5, -5, -100, 300);
  persp.position.set(9, 6, 12);
  let camera = persp, orthoSpan = 12;
  const ambient = new THREE.AmbientLight(0xffffff, 1.5);
  scene3.add(ambient);
  const sun = new THREE.DirectionalLight(0xffffff, 1.6);
  sun.position.set(6, 10, 8);
  scene3.add(sun);

  let controls = null;
  function makeControls() {
    controls?.dispose();
    controls = new OrbitControls(camera, canvas);
    controls.enableDamping = false;
    controls.minDistance = 4;
    controls.maxDistance = 45;
    controls.screenSpacePanning = true;
    controls.addEventListener("change", draw);
    controls.addEventListener("start", () => { userCamera = true; });
  }

  let spec = null, meshes = new Map(), userCamera = false, cssColors = readColors(), concept = false;
  let sel = null, lastState = null, dead = false;
  const labelCache = new Map();
  const linkLines = [];
  const raycaster = new THREE.Raycaster();
  const ndc = new THREE.Vector2();

  function readColors() {
    const cs = getComputedStyle(host);
    const get = (n, d) => (cs.getPropertyValue(n).trim() || d);
    const bg = get("--bg", "#ffffff");
    const n = /^#[0-9a-f]{6}$/i.test(bg) ? parseInt(bg.slice(1), 16) : 0xffffff;
    return { fg: cs.color || "#222", accent: get("--accent", "#0b5cad"), muted: get("--muted", "#666"), line: get("--line", "#bbb"),
             pill: `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, 0.86)` };
  }

  // ---- text sprites
  const FONTS = { sans: "system-ui, 'Segoe UI', sans-serif", mono: "ui-monospace, 'SF Mono', Menlo, Consolas, monospace", serif: "Georgia, 'Times New Roman', serif" };
  // `text` is a string, or a JSON list of parts [{t, c, f, w}] drawn side by side in different colours and fonts.
  function labelTexture(text, color, px = 96, weight = 700, bg = null) {
    const key = `${typeof text === "string" ? text : JSON.stringify(text)}|${color}|${px}|${weight}|${bg}`;
    if (labelCache.has(key)) return labelCache.get(key);
    const c = document.createElement("canvas");
    const ctx = c.getContext("2d");
    const parts = Array.isArray(text) ? text.map((p) => ({ t: p.t, c: p.c === "fg" || !p.c ? color : p.c, font: `${p.w || weight} ${px}px ${FONTS[p.f || "sans"]}` }))
      : String(text).split("\n").map((l) => ({ t: l, c: color, font: `${weight} ${px}px ${FONTS.sans}`, line: true }));
    const lines = Array.isArray(text) ? [parts] : parts.map((p) => [p]);
    const measure = (ln) => ln.reduce((s, p) => { ctx.font = p.font; return s + ctx.measureText(p.t).width; }, 0);
    const w = Math.ceil(Math.max(...lines.map(measure))) + 24;
    const h = Math.ceil(px * 1.25 * lines.length) + 16;
    c.width = w; c.height = h;
    if (bg) { ctx.fillStyle = bg; ctx.beginPath(); ctx.roundRect(2, 2, w - 4, h - 4, 18); ctx.fill(); }
    ctx.textAlign = "left"; ctx.textBaseline = "middle";
    lines.forEach((ln, i) => {
      let x = (w - measure(ln)) / 2;
      for (const p of ln) { ctx.font = p.font; ctx.fillStyle = p.c; ctx.fillText(p.t, x, 8 + px * 1.25 * (i + 0.5)); x += ctx.measureText(p.t).width; }
    });
    const tex = new THREE.CanvasTexture(c);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 4;
    const entry = { tex, aspect: w / h };
    labelCache.set(key, entry);
    return entry;
  }
  function makeSprite() {
    const m = new THREE.SpriteMaterial({ transparent: true, depthWrite: false, depthTest: false });   // numbers stay readable through the glass
    const s = new THREE.Sprite(m);
    s.renderOrder = 10;
    return s;
  }
  function setSpriteText(sprite, text, color, height, px, weight, bg = null) {
    if (!text) { sprite.visible = false; return; }
    const { tex, aspect } = labelTexture(text, color, px, weight, bg);
    if (sprite.material.map !== tex) { sprite.material.map = tex; sprite.material.needsUpdate = true; }
    sprite.scale.set(height * aspect, height, 1);
    sprite.visible = true;
  }

  // ---- object builders
  const boxGeo = new THREE.BoxGeometry(1, 1, 1);
  const roundGeo = new RoundedBoxGeometry(1, 1, 1, 3, 0.07);
  const edgeGeo = new THREE.EdgesGeometry(boxGeo);

  function build(o) {
    const g = new THREE.Group();
    g.userData = { id: o.id, type: o.type };
    if (o.type === "cube") {
      const mat = new THREE.MeshStandardMaterial({ color: o.color || "#0072b2", transparent: true, opacity: 0.4, roughness: 0.55, metalness: 0, depthWrite: false });
      const mesh = new THREE.Mesh(concept ? roundGeo : boxGeo, mat);
      const edges = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: o.edge || o.color || "#0072b2", transparent: true }));
      const sprite = makeSprite();
      g.add(mesh, edges, sprite);
      g.userData.mesh = mesh; g.userData.edges = edges; g.userData.sprite = sprite;
    } else if (o.type === "text") {
      const sprite = makeSprite();
      g.add(sprite); g.userData.sprite = sprite;
    } else if (o.type === "arrow" || o.type === "line") {
      const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 1, 0)]);
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: o.color || "#888", transparent: true }));
      g.add(line); g.userData.line = line;
      if (o.type === "arrow") {
        const head = new THREE.Mesh(new THREE.ConeGeometry(0.11, 0.34, 14), new THREE.MeshBasicMaterial({ color: o.color || "#888", transparent: true }));
        g.add(head); g.userData.head = head;
      }
      const sprite = makeSprite(); g.add(sprite); g.userData.sprite = sprite;
    } else if (o.type === "dim") {
      const geo = new THREE.BufferGeometry().setFromPoints(Array.from({ length: 6 }, () => new THREE.Vector3()));
      const seg = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: o.color || "#6b7680", transparent: true }));
      g.add(seg); g.userData.seg = seg;
      const sprite = makeSprite(); g.add(sprite); g.userData.sprite = sprite;
    } else if (o.type === "box") {
      const edges = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({ color: o.color || "#888", transparent: true }));
      g.add(edges); g.userData.edges = edges;
      const sprite = makeSprite(); g.add(sprite); g.userData.sprite = sprite;
    } else if (o.type === "plane") {
      const mesh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ color: o.color || "#888", transparent: true, side: THREE.DoubleSide, depthWrite: false }));
      g.add(mesh); g.userData.mesh = mesh;
    }
    scene3.add(g);
    return g;
  }

  function disposeAll() {
    for (const g of meshes.values()) {
      scene3.remove(g);
      g.traverse((c) => { if (c.material && c.material !== undefined) c.material.dispose?.(); if (c.geometry && c.geometry !== boxGeo && c.geometry !== edgeGeo) c.geometry.dispose?.(); });
    }
    meshes = new Map();
    linkLines.forEach((l) => { scene3.remove(l); l.geometry.dispose(); l.material.dispose(); });
    linkLines.length = 0;
  }

  function setSpec(next) {
    disposeAll();
    if (next.objects.length > CAP) throw new Error(`Scene has more than ${CAP} objects`);
    spec = next;
    concept = spec.style === "concept";
    ambient.intensity = concept ? 2.1 : 1.5; sun.intensity = concept ? 1.7 : 1.6;
    for (const o of spec.objects) meshes.set(o.id, build(o));
    userCamera = false;
    const wantOrtho = spec.camera?.mode === "ortho";
    orthoSpan = spec.camera?.span || 12;
    if ((camera === ortho) !== wantOrtho || !controls) { camera = wantOrtho ? ortho : persp; makeControls(); }
    resize();
  }

  const V = (a) => new THREE.Vector3(a[0], a[1], a[2]);

  // ---- per-frame update
  function update(t, selection) {
    if (!spec || dead) return;
    sel = selection;
    const state = evalScene(spec, t);
    lastState = state;
    const rel = relatedIds(spec, selection);
    const dimOthers = rel.sel != null;
    cssColors = cssColors || readColors();
    for (const o of spec.objects) {
      const st = state.objects[o.id], g = meshes.get(o.id);
      const involved = !dimOthers || o.id === rel.sel || rel.linked.has(o.id);
      const strong = o.id === rel.sel, linked = rel.linked.has(o.id);
      let op = (st.opacity ?? 1) * (involved ? 1 : 0.22);
      const vis = st.visible !== false && op > 0.01;
      g.visible = vis;
      if (!vis) continue;
      if (st.pos) g.position.set(st.pos[0], st.pos[1], st.pos[2]);
      if (o.type === "cube") {
        const s = (st.size ?? 0.9) * (st.scale ?? 1) * (strong ? 1.1 : 1);
        g.userData.mesh.scale.setScalar(s); g.userData.edges.scale.setScalar(s);
        const mat = g.userData.mesh.material;
        mat.color.set(st.color || "#0072b2"); mat.roughness = concept ? 0.4 : 0.55;
        mat.opacity = Math.min(1, op * (st.ghost ? 0.5 : 1) * (strong || linked ? 1.25 : 1) * (concept ? 0.82 : 0.3));
        mat.emissive.set(strong ? cssColors.accent : linked ? cssColors.accent : "#000000");
        mat.emissiveIntensity = strong ? 0.55 : linked ? 0.3 : 0;
        const em = g.userData.edges.material;
        em.color.set(strong || linked ? cssColors.accent : st.edge || st.color || "#0072b2");
        em.opacity = Math.min(1, op * (st.ghost ? 0.7 : 1));
        setSpriteText(g.userData.sprite, st.label, cssColors.fg, Math.max(0.26, s * (concept ? 0.5 : 0.7)), 96, 800);
        g.userData.sprite.material.opacity = Math.min(1, op);
      } else if (o.type === "text") {
        g.userData.sprite.scale.setScalar(1);
        setSpriteText(g.userData.sprite, st.text, st.color && st.color !== "fg" ? st.color : cssColors.fg, st.height ?? 0.6, 80, st.weight ?? 600, cssColors.pill);
        g.userData.sprite.material.opacity = Math.min(1, op);
      } else if (o.type === "arrow" || o.type === "line") {
        const a = V(st.from), b = V(st.to), p = st.progress ?? 1;
        const end = a.clone().lerp(b, p);
        const pos = g.userData.line.geometry.attributes.position;
        pos.setXYZ(0, a.x, a.y, a.z); pos.setXYZ(1, end.x, end.y, end.z); pos.needsUpdate = true;
        g.userData.line.material.color.set(st.color || "#888"); g.userData.line.material.opacity = op;
        if (g.userData.head) {
          const dir = end.clone().sub(a);
          g.userData.head.visible = dir.length() > 0.05 && p > 0.02;
          if (dir.length() > 0) { g.userData.head.position.copy(end); g.userData.head.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize()); }
          g.userData.head.material.color.set(st.color || "#888"); g.userData.head.material.opacity = op;
        }
        g.position.set(0, 0, 0);
        const lp = end.clone().add(new THREE.Vector3(...(st.label_offset || [0.2, 0.2, 0])));
        g.userData.sprite.position.copy(lp);
        setSpriteText(g.userData.sprite, st.label, st.color && st.color !== "#888" ? st.color : cssColors.fg, 0.5, 72, 700, cssColors.pill);
        g.userData.sprite.material.opacity = op;
      } else if (o.type === "dim") {
        const a = V(st.from), b = V(st.to), tk = V(st.tick || [0, -0.22, 0]);
        const pos = g.userData.seg.geometry.attributes.position;
        const pts = [a, b, a, a.clone().add(tk), b, b.clone().add(tk)];
        pts.forEach((q, i) => pos.setXYZ(i, q.x, q.y, q.z)); pos.needsUpdate = true;
        g.userData.seg.material.color.set(st.color || cssColors.muted); g.userData.seg.material.opacity = op;
        g.position.set(0, 0, 0);
        const mid = a.clone().add(b).multiplyScalar(0.5).add(V(st.label_offset || [0, -0.55, 0]));
        g.userData.sprite.position.copy(mid);
        setSpriteText(g.userData.sprite, st.label, cssColors.muted, st.height ?? 0.4, 64, 500);
        g.userData.sprite.material.opacity = op;
      } else if (o.type === "box") {
        const sz = st.size || [1, 1, 1];
        g.userData.edges.scale.set(sz[0], sz[1], sz[2]);
        g.userData.edges.material.color.set(st.color || cssColors.muted); g.userData.edges.material.opacity = op;
        g.userData.sprite.position.set(0, sz[1] / 2 + 0.55, 0);
        setSpriteText(g.userData.sprite, st.label, cssColors.fg, 0.5, 72, 600, cssColors.pill);
        g.userData.sprite.material.opacity = op;
      } else if (o.type === "plane") {
        const sz = st.size || [1, 1];
        g.userData.mesh.scale.set(sz[0], sz[1], 1);
        g.userData.mesh.rotation.set(...(st.rotation || [0, 0, 0]));
        g.userData.mesh.material.color.set(st.color || "#888"); g.userData.mesh.material.opacity = op;
      }
    }
    // Concept style: a number on a cube hidden behind other solid cubes fades, so that front numbers stay readable.
    if (concept) {
      const solid = [];
      for (const o of spec.objects) {
        const g = meshes.get(o.id), st = state.objects[o.id];
        if (o.type === "cube" && g.visible && (st.opacity ?? 1) > 0.3 && (st.size ?? 0.9) * (st.scale ?? 1) > 0.5) solid.push({ id: o.id, c: g.position, h: (st.size ?? 0.9) * (st.scale ?? 1) / 2 });
      }
      camera.updateMatrixWorld();
      const eye = new THREE.Vector3(); camera.getWorldDirection(eye);
      const ray = new THREE.Ray(), box = new THREE.Box3(), hit = new THREE.Vector3(), tmp = new THREE.Vector3();
      for (const s of solid) {
        const g = meshes.get(s.id);
        const origin = camera.isOrthographicCamera ? tmp.copy(s.c).addScaledVector(eye, -80).clone() : camera.position.clone();
        const dir = s.c.clone().sub(origin); const dist = dir.length(); dir.normalize();
        ray.set(origin, dir);
        let hidden = false;
        for (const o2 of solid) {
          if (o2.id === s.id) continue;
          box.min.set(o2.c.x - o2.h, o2.c.y - o2.h, o2.c.z - o2.h); box.max.set(o2.c.x + o2.h, o2.c.y + o2.h, o2.c.z + o2.h);
          if (ray.intersectBox(box, hit) && origin.distanceTo(hit) < dist - s.h * 0.6) { hidden = true; break; }
        }
        if (hidden && s.id !== rel.sel && !rel.linked.has(s.id)) g.userData.sprite.material.opacity *= 0.13;
      }
    }
    // lines from the selected object to what it is linked with
    const targets = rel.sel ? [...rel.linked].filter((id) => meshes.get(id)?.visible) : [];
    while (linkLines.length < targets.length) {
      const l = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]), new THREE.LineBasicMaterial({ color: cssColors.accent, transparent: true, opacity: 0.9, depthTest: false }));
      l.renderOrder = 20; scene3.add(l); linkLines.push(l);
    }
    linkLines.forEach((l, i) => {
      l.visible = i < targets.length;
      if (l.visible) {
        const p = l.geometry.attributes.position, a = meshes.get(rel.sel).position, b = meshes.get(targets[i]).position;
        p.setXYZ(0, a.x, a.y, a.z); p.setXYZ(1, b.x, b.y, b.z); p.needsUpdate = true;
        l.material.color.set(cssColors.accent);
      }
    });
    if (state.camera && !userCamera) {
      camera.position.set(...state.camera.pos); controls.target.set(...state.camera.target);
      if (camera === ortho && state.camera.zoom) { ortho.zoom = state.camera.zoom; ortho.updateProjectionMatrix(); }
      controls.update();
    }
    draw();
  }

  function draw() { if (!dead) renderer.render(scene3, camera); }

  function setView(view) {   // view: {pos, target} from the scene's named views
    userCamera = true;
    camera.position.set(...view.pos); controls.target.set(...view.target); controls.update(); draw();
  }
  function resetView() { userCamera = false; if (lastState?.camera) { camera.position.set(...lastState.camera.pos); controls.target.set(...lastState.camera.target); controls.update(); } draw(); }

  function resize() {
    const w = host.clientWidth || 600, h = host.clientHeight || 360;
    renderer.setSize(w, h, false);
    canvas.style.width = "100%"; canvas.style.height = "100%";
    persp.aspect = w / h; persp.updateProjectionMatrix();
    const half = orthoSpan / 2;
    ortho.left = -half * (w / h); ortho.right = half * (w / h); ortho.top = half; ortho.bottom = -half; ortho.updateProjectionMatrix();
    draw();
  }
  const ro = new ResizeObserver(resize); ro.observe(host);
  makeControls();
  resize();

  // ---- picking
  function pickAt(clientX, clientY) {
    const r = canvas.getBoundingClientRect();
    ndc.set(((clientX - r.left) / r.width) * 2 - 1, -(((clientY - r.top) / r.height) * 2 - 1));
    raycaster.setFromCamera(ndc, camera);
    const targets = [];
    for (const o of spec.objects) {
      const g = meshes.get(o.id);
      if (o.type === "cube" && g.visible && o.pickable !== false && g.userData.mesh.material.opacity > 0.04) targets.push(g.userData.mesh);
    }
    const hit = raycaster.intersectObjects(targets, false)[0];
    return hit ? hit.object.parent.userData.id : null;
  }
  let down = null;
  canvas.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY }; });
  canvas.addEventListener("pointerup", (e) => {
    if (!down || !spec) return;
    const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
    down = null;
    if (moved < 5) onPick?.(pickAt(e.clientX, e.clientY));
  });
  let hoverId = null;
  canvas.addEventListener("pointermove", (e) => {
    if (e.buttons || !spec) return;
    const id = pickAt(e.clientX, e.clientY);
    if (id !== hoverId) { hoverId = id; canvas.style.cursor = id ? "pointer" : "grab"; onHover?.(id); }
  });
  canvas.style.cursor = "grab";

  const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
  const onTheme = () => { cssColors = readColors(); labelCache.clear(); if (spec && lastState) update(lastTime(), sel); };
  let lastT = 0;
  const lastTime = () => lastT;
  mq?.addEventListener?.("change", onTheme);

  return {
    setSpec,
    update(t, s) { lastT = t; update(t, s); },
    setView, resetView, resize,
    get userCamera() { return userCamera; },
    // For tests and for keyboard users: where an object appears on the page.
    project(id) {
      const g = meshes.get(id); if (!g) return null;
      const v = g.position.clone().project(camera), r = canvas.getBoundingClientRect();
      return { x: r.left + ((v.x + 1) / 2) * r.width, y: r.top + ((1 - v.y) / 2) * r.height, visible: g.visible };
    },
    dispose() {
      dead = true; ro.disconnect(); mq?.removeEventListener?.("change", onTheme);
      disposeAll(); labelCache.forEach((e) => e.tex.dispose()); controls?.dispose(); renderer.dispose(); canvas.remove();
    },
  };
}
