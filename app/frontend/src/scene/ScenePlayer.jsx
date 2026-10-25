import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Inline } from "../Inline.jsx";
import { chapterIndex, fmtTime, relatedIds } from "./timeline.js";
import { createSceneRenderer } from "./sceneRenderer.js";

const HELP_LABEL = {
  "smaller-example": "A smaller example", prerequisite: "Prerequisite", geometry: "Geometry",
  "expanded-equation": "The equation with numbers", "common-mistake": "A common mistake", "code-connection": "Connect to the code",
};
const SPEEDS = [0.5, 1, 1.5, 2];
const paramKey = (params, values) => params.map((p) => String(values[p.name])).join("|");

function useReducedMotion() {
  const [r, setR] = useState(() => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false);
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    const on = (e) => setR(e.matches);
    mq?.addEventListener("change", on);
    return () => mq?.removeEventListener("change", on);
  }, []);
  return r;
}

export default function ScenePlayer({ id, demo, caption }) {
  const data = demo.data;
  const params = data.params || [];
  const [values, setValues] = useState(demo.input_schema.defaults || {});
  const scene = data.cases[paramKey(params, values)].scene;
  const reduced = useReducedMotion();

  const host = useRef(null);
  const api = useRef(null);
  const raf = useRef(0);
  const clock = useRef({ t: 0, last: 0, until: null, playing: false, speed: 1 });
  const [t, setT] = useState(scene.start_t ?? scene.chapters[0].t1);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [selection, setSelection] = useState(null);
  const [hover, setHover] = useState(null);
  const [status, setStatus] = useState("loading");     // loading | ready | nowebgl
  const [helpOpen, setHelpOpen] = useState(false);
  const selRef = useRef(null);
  selRef.current = selection;
  const sceneRef = useRef(scene);
  sceneRef.current = scene;

  const byId = useMemo(() => new Map(scene.objects.map((o) => [o.id, o])), [scene]);
  const ch = chapterIndex(scene, t);
  const chapter = scene.chapters[ch];

  const draw = useCallback((time) => { api.current?.update(time, selRef.current); }, []);

  // create the renderer once the stage is visible on screen
  useEffect(() => {
    let cancelled = false, io, started = false;
    const start = async () => {
      if (started) return; started = true;
      try {
        const r = await createSceneRenderer(host.current, {
          onPick: (pid) => setSelection((cur) => (pid && pid !== cur ? pid : null)),
          onHover: (pid) => setHover(pid),
        });
        if (cancelled) { r.dispose(); return; }
        api.current = r;
        host.current.__scene = r;     // read by browser tests
        r.setSpec(sceneRef.current);
        r.update(clock.current.t, selRef.current);
        setStatus("ready");
      } catch (e) {
        console.warn("3D scene unavailable:", e);
        setStatus("nowebgl");
      }
    };
    if ("IntersectionObserver" in window) { io = new IntersectionObserver((es) => es.some((e) => e.isIntersecting) && start(), { rootMargin: "300px" }); io.observe(host.current); }
    else start();
    return () => { cancelled = true; io?.disconnect(); cancelAnimationFrame(raf.current); api.current?.dispose(); api.current = null; };
  }, []);

  // new scene (parameter changed)
  useEffect(() => {
    clock.current.t = scene.start_t ?? scene.chapters[0].t1; clock.current.until = null;
    setT(clock.current.t); setPlaying(false); setSelection(null);
    if (api.current) { api.current.setSpec(scene); api.current.update(clock.current.t, null); }
  }, [scene]);

  useEffect(() => { draw(t); }, [selection]);  // eslint-disable-line react-hooks/exhaustive-deps

  // animation loop
  useEffect(() => {
    clock.current.playing = playing; clock.current.speed = speed;
    if (!playing || reduced) return;
    clock.current.last = performance.now();
    const tick = (now) => {
      const c = clock.current;
      const dt = Math.min((now - c.last) / 1000, 0.1) * c.speed;
      c.last = now;
      const limit = c.until ?? sceneRef.current.duration;
      c.t = Math.min(c.t + dt, limit);
      setT(c.t); draw(c.t);
      if (c.t >= limit - 1e-6) { c.until = null; setPlaying(false); return; }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, speed, reduced, draw]);

  const seek = useCallback((time, { play = false, until = null } = {}) => {
    const c = clock.current;
    c.t = Math.min(Math.max(time, 0), sceneRef.current.duration); c.until = until;
    setT(c.t); draw(c.t); setPlaying(play && !reduced);
  }, [draw, reduced]);

  const playChapter = (i) => {
    const c = scene.chapters[i];
    api.current?.resetView();
    if (reduced) seek(c.t1 - 0.001); else seek(c.t0, { play: true, until: c.t1 - 0.001 });
  };
  const togglePlay = () => {
    const c = clock.current;
    if (playing) { setPlaying(false); return; }
    if (c.t >= scene.duration - 0.01) c.t = 0;
    c.until = null; setPlaying(true);
  };

  const onKeyDown = (e) => {
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
    if (e.key === " " && e.target === host.current) { e.preventDefault(); togglePlay(); }
    if (e.key === "ArrowRight") { e.preventDefault(); playChapter(Math.min(ch + 1, scene.chapters.length - 1)); }
    if (e.key === "ArrowLeft") { e.preventDefault(); playChapter(Math.max(ch - 1, 0)); }
    if (e.key === "Home") { e.preventDefault(); seek(0); }
    if (e.key === "Escape") setSelection(null);
  };

  const trace = selection ? scene.by_sel?.[selection] : null;
  const calc = (selection && scene.calc?.[selection]) || (scene.calc && scene.calc_default ? scene.calc[scene.calc_default] : null);
  const selectable = scene.objects.filter((o) => o.title);
  const help = (demo.help || []).filter((h) => h.step_id === chapter.id);
  const rel = relatedIds(scene, selection);

  return (
    <section className={`demo scenedemo ${scene.style === "concept" ? "concept" : ""}`} id={id} aria-label={caption || "3D animation"} onKeyDown={onKeyDown}>
      {caption && <h3 className="demo-title">{caption}</h3>}
      {params.length > 0 && (
        <fieldset className="inputs"><legend>Change the example</legend>
          <div className="shape-row">
            {params.map((p) => (
              <label key={p.name}>{p.label}
                <select value={String(values[p.name])} onChange={(e) => setValues((v) => ({ ...v, [p.name]: p.options.find((o) => String(o.value) === e.target.value).value }))}>
                  {p.options.map((o) => <option key={String(o.value)} value={String(o.value)}>{o.label}</option>)}
                </select>
              </label>
            ))}
          </div>
        </fieldset>
      )}

      <div className="stagewrap">
        <div ref={host} className="stage" tabIndex={0} role="group" aria-label={`3D animation. ${scene.alt} Press Space to play or pause. Arrow keys change chapter.`} data-ready={status} />
        {status === "loading" && <p className="stagemsg">Loading the 3D scene…</p>}
        {status === "nowebgl" && <p className="stagemsg" role="alert">3D drawing is not available in this browser. Read the text version below.</p>}
        <div className="stagehint" aria-hidden="true">Drag to rotate · scroll to zoom · click a cube</div>
        {hover && byId.get(hover)?.title && <div className="stagetip" aria-hidden="true">{byId.get(hover).title}</div>}
      </div>

      <div className="viewbar" role="group" aria-label="Camera">
        <button type="button" onClick={() => api.current?.resetView()}>Follow the animation</button>
        {Object.entries(scene.views || {}).map(([name, v]) => <button key={name} type="button" onClick={() => api.current?.setView(v)}>{v.label || name}</button>)}
        {selection && <button type="button" onClick={() => setSelection(null)}>Clear selection</button>}
      </div>

      <div className="controlrow">
        <div className="transport" role="group" aria-label="Playback">
          <button type="button" className="play primary" onClick={togglePlay} disabled={reduced} aria-label={playing ? "Pause" : "Play"}>{playing ? "⏸ Pause" : "▶ Play"}</button>
          <button type="button" onClick={() => playChapter(Math.max(ch - 1, 0))} disabled={ch === 0} aria-label="Previous chapter">⏮ Back</button>
          <button type="button" onClick={() => playChapter(Math.min(ch + 1, scene.chapters.length - 1))} disabled={ch === scene.chapters.length - 1} aria-label="Next chapter">⏭ Next</button>
          <button type="button" onClick={() => { seek(scene.start_t ?? scene.chapters[0].t1); setSelection(null); api.current?.resetView(); }} aria-label="Reset">↻ Reset</button>
          <label className="speed">Speed
            <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>{SPEEDS.map((s) => <option key={s} value={s}>{s}×</option>)}</select>
          </label>
          <span className="time" aria-hidden="true">{fmtTime(t)} / {fmtTime(scene.duration)}</span>
        </div>
        <ol className="stepper" aria-label="Chapters">
          {scene.chapters.map((c, i) => (
            <li key={c.id} className={i < ch ? "done" : ""}>
              <button type="button" aria-current={i === ch ? "step" : undefined} onClick={() => playChapter(i)}><span className="dot">{i + 1}</span><span className="steplabel">{c.short || c.title}</span></button>
            </li>
          ))}
        </ol>
      </div>
      {reduced && <p className="hint">Playback is off because your system asks for reduced motion. Use the chapter buttons to jump between chapters.</p>}
      <div className="scrubwrap">
        <input type="range" className="scrub" min={0} max={Math.round(scene.duration * 100)} step={1} value={Math.round(t * 100)}
          aria-label="Timeline position" aria-valuetext={`${fmtTime(t)} of ${fmtTime(scene.duration)}, chapter ${ch + 1}: ${chapter.title}`}
          onChange={(e) => seek(Number(e.target.value) / 100)} />
        <div className="ticks" aria-hidden="true">{scene.chapters.map((c) => <span key={c.id} style={{ left: `${(c.t0 / scene.duration) * 100}%` }} />)}</div>
      </div>
      <div className="explain">
      <div className="steptext" role="status" aria-live="polite">
        <h4>Chapter {ch + 1} of {scene.chapters.length}: {chapter.title}</h4>
        <dl>
          <dt>What changed</dt><dd><Inline text={chapter.changed} /></dd>
          <dt>Why</dt><dd><Inline text={chapter.why} /></dd>
          <dt>What stayed the same</dt><dd><Inline text={chapter.fixed} /></dd>
        </dl>
      </div>
      {calc && (
        <div className="calccard" aria-live="polite">
          <h4>{selection ? "Selected cell computation" : "Example cell computation"}</h4>
          <div className="calcrow">
            {calc.terms.map((term, i) => (
              <span key={i} className="calcpart">
                <span className={`calcbox tone-${term.tone}`}><small className="calclabel">{term.label}</small><b>{term.value}</b></span>
                {calc.ops[i] && <span className="calcop">{calc.ops[i]}</span>}
              </span>
            ))}
          </div>
        </div>
      )}
      </div>
      {selection && (
        <div className="trace" role="status" aria-live="polite">
          <strong>Selected {selection}.</strong> {trace ? <Inline text={trace} /> : null}
          {rel.linked.size > 0 && <span className="sub"> Linked: {[...rel.linked].slice(0, 8).join(", ")}{rel.linked.size > 8 ? "…" : ""}</span>}
        </div>
      )}

      {help.length > 0 && (
        <div className="help">
          <button type="button" aria-expanded={helpOpen} aria-controls={`${id}-help`} onClick={() => setHelpOpen((v) => !v)}>Help me understand this step</button>
          {helpOpen && (
            <div id={`${id}-help`} className="helppanel">
              {help.map((h, n) => (
                <section key={n}><h5>{HELP_LABEL[h.help_type]}</h5><p><Inline text={h.content} /></p>
                  {h.prerequisite_id && <p><a href={`#/article/${h.prerequisite_id}`}>Read: {h.prerequisite_title}</a></p>}</section>
              ))}
            </div>
          )}
        </div>
      )}

      <details className="static">
        <summary>Text version and keyboard selection (no 3D needed)</summary>
        <p>{scene.alt}</p>
        <ol>{scene.chapters.map((c) => <li key={c.id}><strong>{c.title}.</strong> <Inline text={c.changed} /> <Inline text={c.why} /></li>)}</ol>
        {selectable.length > 0 && (
          <>
            <p>Select a number to see its explanation:</p>
            <div className="picklist">
              {selectable.map((o) => (
                <button key={o.id} type="button" aria-pressed={selection === o.id} onClick={() => setSelection(selection === o.id ? null : o.id)} title={o.title}>{o.id}</button>
              ))}
            </div>
          </>
        )}
      </details>
    </section>
  );
}
