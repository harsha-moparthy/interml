import { Suspense, lazy, useEffect, useMemo, useReducer, useState } from "react";
import { Inline } from "../Inline.jsx";
import { COMPONENTS } from "./components.js";
import { initialState, makeReducer } from "./engine.js";

const HELP_LABEL = {
  "smaller-example": "A smaller example",
  prerequisite: "Prerequisite",
  geometry: "Geometry",
  "expanded-equation": "The equation with numbers",
  "common-mistake": "A common mistake",
  "code-connection": "Connect to the code",
};

function useReducedMotion() {
  const [reduced, setReduced] = useState(() => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false);
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    if (!mq) return;
    const on = (e) => setReduced(e.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return reduced;
}

const ScenePlayer = lazy(() => import("../scene/ScenePlayer.jsx"));   // three.js loads only on pages with a 3D scene

export function DemoBlock({ id, demo, caption }) {
  if (demo?.component === "scene") return <Suspense fallback={<p role="status">Loading the 3D animation…</p>}><ScenePlayer id={id} demo={demo} caption={caption} /></Suspense>;
  const entry = demo && COMPONENTS[demo.component];
  if (!entry) return <p role="alert" className="error">Demonstration “{id}” is not available.</p>;
  return <DemoShell id={id} demo={demo} entry={entry} caption={caption} />;
}

function DemoShell({ id, demo, entry, caption }) {
  const { View, Inputs } = entry;
  const spec = useMemo(() => (entry.makeSpec ? entry.makeSpec(demo) : entry.spec), [entry, demo]);
  const defaults = demo.input_schema.defaults;
  const reducer = useMemo(() => makeReducer(spec, defaults), [spec, defaults]);
  const [state, dispatch] = useReducer(reducer, undefined, () => initialState(spec, defaults));
  const reduced = useReducedMotion();
  const { frames } = spec.compute(state.inputs, state.selection);
  const frame = frames[state.step];
  const last = frames.length - 1;
  const stepTitle = demo.steps.find((s) => s.id === frame.id)?.title || frame.title;
  const help = demo.help.filter((h) => h.step_id === frame.id);

  // Play advances one step at a time. It never starts by itself and is off under reduced motion.
  useEffect(() => {
    if (!state.playing || reduced) return;
    const t = setInterval(() => dispatch({ type: "next" }), 1600);
    return () => clearInterval(t);
  }, [state.playing, reduced]);

  const onKeyDown = (e) => {
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
    if (e.key === "ArrowRight") { e.preventDefault(); dispatch({ type: "next" }); }
    if (e.key === "ArrowLeft") { e.preventDefault(); dispatch({ type: "prev" }); }
    if (e.key === "Home") { e.preventDefault(); dispatch({ type: "goto", step: 0 }); }
  };

  const concept = demo.component === "concept-lab";
  return (
    <section className={`demo ${concept ? "concept-demo" : ""}`} id={id} aria-label={caption || "Interactive demonstration"} onKeyDown={onKeyDown}>
      {concept && <header className="demo-head"><div><h3>{caption || "Interactive concept laboratory"}</h3><p>Follow the objects, then change one condition</p></div><div className="demo-progress"><span>{state.step + 1} / {frames.length}</span><i style={{"--progress": `${((state.step + 1) / frames.length) * 100}%`}} /></div></header>}
      {!concept && caption && <h3 className="demo-title">{caption}</h3>}

      <View frame={frame} state={state} dispatch={dispatch} demo={demo} demoId={id} />

      {concept ? <div className="cl-controls">
        <div className="cl-buttons" role="group" aria-label="Playback controls">
          {state.playing
            ? <button className="cl-play" type="button" onClick={() => dispatch({ type: "pause" })}>Ⅱ Pause</button>
            : <button className="cl-play" type="button" onClick={() => state.step >= last ? dispatch({type:"goto",step:0}) : dispatch({ type: "play" })} disabled={reduced}>▶ Play</button>}
          <button type="button" onClick={() => dispatch({ type: "prev" })} disabled={state.step === 0}>← Back</button>
          <button type="button" onClick={() => dispatch({ type: "next" })} disabled={state.step >= last}>Next →</button>
          <button type="button" onClick={() => dispatch({ type: "reset" })}>↺ Reset</button>
        </div>
        <ol className="cl-steps" aria-label="Animation chapters">{frames.map((f,n)=><li key={f.id}><button type="button" aria-current={n===state.step?"step":undefined} onClick={()=>dispatch({type:"goto",step:n})}><span>0{n+1}</span>{demo.steps.find((s)=>s.id===f.id)?.title||f.title}</button></li>)}</ol>
        <input className="cl-scrub" aria-label="Animation timeline" type="range" min="0" max={last} step="1" value={state.step} onChange={(e)=>dispatch({type:"goto",step:Number(e.target.value)})}/>
      </div> : <>
        {Inputs && <Inputs inputs={state.inputs} dispatch={dispatch} demo={demo} />}
        <ol className="timeline" aria-label="Steps">{frames.map((f,n)=><li key={f.id}><button type="button" aria-current={n===state.step?"step":undefined} onClick={()=>dispatch({type:"goto",step:n})}><span className="num">{n+1}</span> {demo.steps.find((s)=>s.id===f.id)?.title||f.title}</button></li>)}</ol>
        <div className="controls" role="group" aria-label="Playback controls"><button type="button" onClick={()=>dispatch({type:"reset"})}>Reset</button><button type="button" onClick={()=>dispatch({type:"prev"})} disabled={state.step===0}>◀ Back</button>{state.playing?<button type="button" onClick={()=>dispatch({type:"pause"})}>Pause</button>:<button type="button" onClick={()=>dispatch({type:"play"})} disabled={reduced||state.step>=last}>Play</button>}<button type="button" onClick={()=>dispatch({type:"next"})} disabled={state.step>=last}>Forward ▶</button></div>
      </>}

      {concept ? <div className="cl-caption" role="status" aria-live="polite"><div><h4>Step {state.step + 1}: {stepTitle}</h4><dl><dt>Changed</dt><dd><Inline text={frame.changed}/></dd><dt>Why</dt><dd><Inline text={frame.why}/></dd></dl></div><div className="cl-invariant"><span>Held constant</span><p><Inline text={frame.fixed}/></p></div></div>
        : <div className="steptext" role="status" aria-live="polite"><h4>Step {state.step + 1} of {frames.length}: {stepTitle}</h4><dl><dt>What changed</dt><dd><Inline text={frame.changed}/></dd><dt>Why</dt><dd><Inline text={frame.why}/></dd><dt>What stayed the same</dt><dd><Inline text={frame.fixed}/></dd></dl></div>}

      {concept && Inputs && <details className="cl-explore"><summary>Explore a different case</summary><Inputs inputs={state.inputs} dispatch={dispatch} demo={demo}/></details>}

      {help.length > 0 && (
        <div className="help">
          <button type="button" aria-expanded={state.helpOpen} aria-controls={`${id}-help`} onClick={() => dispatch({ type: "toggleHelp" })}>
            Help me understand this step
          </button>
          {state.helpOpen && (
            <div id={`${id}-help`} className="helppanel">
              {help.map((h, n) => (
                <section key={n}>
                  <h5>{HELP_LABEL[h.help_type]}</h5>
                  <p><Inline text={h.content} /></p>
                  {h.prerequisite_id && <p><a href={`#/article/${h.prerequisite_id}`}>Read: {h.prerequisite_title}</a></p>}
                </section>
              ))}
            </div>
          )}
        </div>
      )}

      <details className="static">
        <summary>Text version of every step (no animation)</summary>
        <ol>
          {frames.map((f) => (
            <li key={f.id}><strong>{demo.steps.find((s) => s.id === f.id)?.title || f.title}.</strong>{" "}
              <Inline text={f.changed} /> <Inline text={f.why} /></li>
          ))}
        </ol>
      </details>
    </section>
  );
}

export function CompareBlock({ block, demo }) {
  const entry = demo && COMPONENTS[demo.component];
  const spec = useMemo(() => (entry?.makeSpec ? entry.makeSpec(demo) : entry?.spec), [entry, demo]);
  if (!entry || (!entry.CompareView && !entry.makeSpec)) return <p role="alert" className="error">Comparison is not available.</p>;
  const { CompareView, View } = entry;
  const noop = () => {};
  return (
    <section className="compare" aria-label={`Comparison: ${block.changed}`}>
      <h3>Compare two cases</h3>
      <p>Only one condition changes: <strong><Inline text={block.changed} /></strong>.</p>
      <div className="cases">
        {block.cases.map((c) => {
          const inputs = spec.sanitize(c.inputs);
          const computed = spec.compute(inputs, spec.defaultSelection(inputs));
          const frame = block.frame ? computed.frames.find((f) => f.id === block.frame) || computed.frames.at(-1) : computed.frames.at(-1);
          return (
            <article key={c.label}>
              <h4>{c.label}</h4>
              {CompareView
                ? <CompareView inputs={inputs} computed={computed} />
                : <>
                    <p className="ftext"><Inline text={frame.changed} /></p>
                    <View frame={frame} state={{ selection: null, inputs }} dispatch={noop} demo={demo} />
                  </>}
            </article>
          );
        })}
      </div>
      <p className="takeaway"><Inline text={block.takeaway} /></p>
    </section>
  );
}
