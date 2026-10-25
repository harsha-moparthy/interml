import { useEffect, useMemo, useState } from "react";
import { Inline } from "../Inline.jsx";
import { pad } from "../lib.js";
import { apiFetch } from "../staticApi.js";
import { CodeView } from "./CodeView.jsx";

/** The supplied solution with authored notes. Each note marks a line range and explains it.
 * The solution stays hidden until the learner asks for it, as on the question page. */
export function WalkthroughBlock({ exercise, title, notes }) {
  const [shown, setShown] = useState(false);
  const [sol, setSol] = useState({ data: null, error: null });
  const [pinned, setPinned] = useState(null);   // chosen by click or key
  const [hover, setHover] = useState(null);     // previewed while the pointer or focus is on a note
  const active = hover ?? pinned;
  const setActive = (id) => setPinned((p) => (p === id ? null : id));

  useEffect(() => {
    if (!shown || sol.data) return;
    let live = true;
    apiFetch(`/api/exercises/${exercise}/solution`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status} ${r.statusText}`))))
      .then((data) => live && setSol({ data, error: null }))
      .catch((error) => live && setSol({ data: null, error }));
    return () => { live = false; };
  }, [shown, exercise, sol.data]);

  const highlights = useMemo(() => notes.map((n, i) => ({ id: String(i), from: n.lines[0], to: n.lines[1], kind: "note" })), [notes]);

  return (
    <section className="walkthrough" aria-label={title || `Solution walkthrough for question ${pad(exercise)}`}>
      <h3>{title || `Read the supplied solution for question ${pad(exercise)}`}</h3>
      <button type="button" aria-expanded={shown} onClick={() => setShown((v) => !v)}>
        {shown ? "Hide solution and notes" : "Show solution with notes"}
      </button>
      {shown && !sol.data && !sol.error && <p role="status">Loading solution…</p>}
      {sol.error && <p role="alert" className="error">Cannot load the solution: {sol.error.message}</p>}
      {shown && sol.data && (
        <div className="walk">
          <ol className="notes">
            {notes.map((n, i) => (
              <li key={i}>
                <button type="button" aria-pressed={pinned === String(i)} onClick={() => setActive(String(i))}
                  onMouseEnter={() => setHover(String(i))} onMouseLeave={() => setHover(null)}
                  onFocus={() => setHover(String(i))} onBlur={() => setHover(null)}>
                  <span className="num">{i + 1}</span> <span className="lines">line{n.lines[0] === n.lines[1] ? ` ${n.lines[0]}` : `s ${n.lines[0]}–${n.lines[1]}`}</span>
                  <span className="text"><Inline text={n.text} /></span>
                </button>
              </li>
            ))}
          </ol>
          <CodeView code={sol.data.code} language={sol.data.language} highlights={highlights} activeId={active}
            onHighlightClick={(id) => setPinned(id)} label={`Solution code for question ${pad(exercise)}`} maxHeight="30rem" />
        </div>
      )}
    </section>
  );
}
