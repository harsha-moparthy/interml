import { useEffect, useRef, useState } from "react";
import { Inline } from "../Inline.jsx";
import { CodeEditor } from "../code/CodeView.jsx";
import { pad } from "../lib.js";
import { sharedRunner } from "./runner.js";

// Uncompressed download sizes in MB, from `npm run setup:pyodide`. Pandas includes its three small dependencies.
const PACKAGE_MB = { numpy: 3.1, pandas: 6.4 };
const VERSIONS = { numpy: "NumPy 2.0.2", pandas: "pandas 2.2.3" };
const CORE_MB = 14;

function runtimeNote(packages) {
  const names = packages.map((p) => VERSIONS[p] || p).join(", ");
  const mb = Math.round(CORE_MB + packages.reduce((s, p) => s + (PACKAGE_MB[p] || 0), 0));
  return `Runs in your browser with Pyodide 0.27.7 (Python 3.12) and ${names}. Browser memory limits apply: ` +
    `WebAssembly memory cannot exceed 4 GB, and your browser may allow less. The first run downloads about ${mb} MB ` +
    "(uncompressed); your browser keeps it for later visits.";
}

export function PlaygroundBlock({ exercise, pg }) {
  if (!pg) return <p role="alert" className="error">The playground for question {pad(exercise)} is not available.</p>;
  return pg.runtime === "display-only" ? <DisplayOnly exercise={exercise} pg={pg} /> : <Runnable exercise={exercise} pg={pg} />;
}

// While a run is active, the shared runtime state says whether it is still loading or already running.
const busyPhase = (phase, runtime) => (phase === "loading" || phase === "running" ? (runtime === "running" ? "running" : "loading") : phase);

function DisplayOnly({ exercise, pg }) {
  return (
    <section className="playground display-only" aria-label={`Code for question ${pad(exercise)}, not run in the browser`}>
      <h3>Question {pad(exercise)} code <span className="badge planned">Display only</span></h3>
      <p><Inline text={pg.note} /></p>
      {pg.starter_code && <pre tabIndex={0}><code>{pg.starter_code}</code></pre>}
      <p className="sub inline">The supplied solution is on <a href={`#/question/${exercise}`}>the question page</a>.</p>
    </section>
  );
}

function Runnable({ exercise, pg }) {
  const runner = sharedRunner();
  const [code, setCode] = useState(pg.starter_code);
  const [phase, setPhase] = useState("idle");   // idle | loading | running | done | failed | stopped | timeout | unavailable
  const [detail, setDetail] = useState("");
  const [out, setOut] = useState([]);
  const [error, setError] = useState(null);
  const [checks, setChecks] = useState([]);
  const [runtime, setRuntime] = useState(runner.state);
  const active = useRef(false);
  const limitS = pg.time_limit_ms / 1000;

  useEffect(() => runner.subscribe((s) => setRuntime(s.state)), [runner]);
  useEffect(() => () => { if (active.current) runner.stop(); }, [runner]);   // leaving the page stops a running program

  const clear = () => { setOut([]); setError(null); setChecks([]); setDetail(""); };

  async function run() {
    if (active.current) return;
    clear();
    active.current = true;
    setPhase("loading");
    const res = await runner.run(code, {
      checks: pg.checks, timeoutMs: pg.time_limit_ms, packages: pg.packages,
      onOut: (m) => setOut((o) => [...o, { stream: m.stream, text: m.text }]),
    });
    active.current = false;
    if (res.status === "done") {
      setError(res.error); setChecks(res.checks);
      setDetail(`${res.ms} ms`);
      setPhase(res.ok ? "done" : "failed");
    } else if (res.status === "stopped") setPhase("stopped");
    else if (res.status === "timeout") setPhase("timeout");
    else if (res.status === "crashed") setPhase("crashed");
    else { setPhase("unavailable"); setDetail(res.message || ""); }
  }

  const stop = () => { runner.stop(); };
  const reset = () => { runner.stop(); active.current = false; setCode(pg.starter_code); clear(); setPhase("idle"); };

  // phase follows the runtime while the first run loads it
  const shown = busyPhase(phase, runtime);
  const STATUS = {
    idle: "Not started. The Python runtime loads when you press Run.",
    loading: runtime === "ready" || runtime === "idle" ? "Starting…" : "Loading the Python runtime and packages…",
    running: `Running. The time limit is ${limitS} s.`,
    done: `Finished in ${detail}.`,
    failed: `Stopped by an error after ${detail}.`,
    stopped: "Stopped. A fresh runtime starts for the next run.",
    timeout: `Stopped after the ${limitS} s time limit. A fresh runtime starts for the next run.`,
    crashed: "The Python runtime stopped unexpectedly, most likely because it ran out of memory. A fresh runtime starts for the next run.",
    unavailable: `The Python runtime could not start${detail ? `: ${detail}` : ""}. Run “npm run setup:pyodide” in app/frontend and reload.`,
  };
  const busy = shown === "loading" || shown === "running";
  const failedChecks = checks.filter((c) => !c.ok).length;

  return (
    <section className="playground" aria-label={`Python playground for question ${pad(exercise)}`}>
      <h3>Try it: question {pad(exercise)}</h3>
      {pg.note && <p><Inline text={pg.note} /></p>}
      {pg.max_input_size && <p className="sub inline">Keep inputs to {pg.max_input_size} values or fewer.</p>}
      <div className="editor-label" id={`pg-${exercise}-label`}>Python code <span className="sub inline">(Ctrl+Enter runs. Esc then Tab leaves the editor.)</span></div>
      <CodeEditor value={code} onChange={setCode} onRun={run} label={`Python code for question ${pad(exercise)}`} />
      <div className="controls" role="group" aria-label="Run controls">
        <button type="button" onClick={run} disabled={busy}>Run</button>
        <button type="button" onClick={stop} disabled={!busy}>Stop</button>
        <button type="button" onClick={reset}>Reset code and output</button>
      </div>
      <p id={`pg-${exercise}-status`} role="status" aria-live="polite" className={`pgstatus ${shown}`}>{STATUS[shown]}</p>

      {(out.length > 0 || error) && (
        <>
          <h4>Output</h4>
          <pre className="output" tabIndex={0}>{out.map((o, n) => <span key={n} className={o.stream}>{o.text}</span>)}{error && <span className="stderr">{error}</span>}</pre>
        </>
      )}
      {checks.length > 0 && (
        <div className="checks-result">
          <h4>Checks: {failedChecks === 0 ? "all passed" : `${failedChecks} of ${checks.length} failed`}</h4>
          <ul>{checks.map((c) => <li key={c.name}><strong>{c.ok ? "Pass" : "Fail"}</strong> · {c.name}{!c.ok && c.error && <> — <code>{c.error}</code></>}</li>)}</ul>
        </div>
      )}
      <p className="sub">{runtimeNote(pg.packages)}</p>
    </section>
  );
}
