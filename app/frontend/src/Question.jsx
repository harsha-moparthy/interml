import { useEffect, useMemo, useState } from "react";
import { CodeView } from "./code/CodeView.jsx";
import { Inline, Tex } from "./Inline.jsx";
import { useFetch, useAllQuestions, pad } from "./lib.js";
import { apiFetch } from "./staticApi.js";
import { progress } from "./progress.js";
import { useProgress } from "./useProgress.js";
import { useShortcuts } from "./shortcuts.js";
import { ErrorNotice, Skeleton } from "./ui.jsx";

const LESSON_LABEL = { complete: "Deep lesson available", planned: "Deep lesson planned", none: "" };
const PURPOSE_LABEL = { "deep-lesson": "Deep lesson", prerequisite: "Prerequisite article" };
const STATE_LABEL = { draft: "Draft", reviewed: "Reviewed", complete: "Complete" };
const SOURCE_LABEL = { DM: "Deep-ML", TT: "TensorTonic", RESUME: "Resume project", EXT: "External", ADDL: "Additional interview set" };

function SourceLink({ q }) {
  return q.source_url ? (
    <>
      <a href={q.source_url} target="_blank" rel="noopener noreferrer">Original source ({SOURCE_LABEL[q.source_key] || q.source_key})</a>
      {" "}<span className="sub inline">{q.source_url}</span>
    </>
  ) : null;
}

export function MarkdownPrompt({ text }) {
  const parts = text.split(/(\$\$[\s\S]*?\$\$)/g);
  return parts.map((part, i) => part.startsWith("$$")
    ? <div className="prompt-equation" key={i}><Tex tex={part.slice(2, -2).trim()} display /></div>
    : part.split(/\n\s*\n/).filter((p) => p.trim()).map((p, j) => <p key={`${i}-${j}`}><Inline text={p.replace(/\n/g, " ")} /></p>)
  );
}

function Pager({ q, position }) {
  return (
    <nav className="pager qpager" aria-label="Question navigation">
      <a href="#/">← All questions</a>
      {q.previous_id ? <a href={`#/question/${q.previous_id}`} rel="prev">← {pad(q.previous_id)}</a> : <span className="pager-off" aria-disabled="true">← first</span>}
      {q.next_id ? <a href={`#/question/${q.next_id}`} rel="next">{pad(q.next_id)} →</a> : <span className="pager-off" aria-disabled="true">last →</span>}
      {position && <span className="pager-pos">{position.at} of {position.of} in {q.chapter_id}</span>}
    </nav>
  );
}

export function QuestionPage({ id }) {
  const { data: q, error, retry } = useFetch(`/api/exercises/${id}`);
  const all = useAllQuestions();
  const prog = useProgress();
  const [shown, setShown] = useState(false);
  const [sol, setSol] = useState({ data: null, error: null });
  const [attempt, setAttempt] = useState(0);

  // Each question starts with the solution hidden.
  useEffect(() => { setShown(false); setSol({ data: null, error: null }); window.scrollTo(0, 0); progress.touch("question", id); }, [id]);
  useEffect(() => {
    if (!shown || sol.data) return;
    let live = true;
    apiFetch(`/api/exercises/${id}/solution`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status} ${r.statusText}`))))
      .then((data) => live && setSol({ data, error: null }))
      .catch((e) => live && setSol({ data: null, error: e }));
    return () => { live = false; };
  }, [shown, id, sol.data, attempt]);

  useShortcuts({
    ArrowLeft: () => { if (!q?.previous_id) return false; window.location.hash = `#/question/${q.previous_id}`; },
    ArrowRight: () => { if (!q?.next_id) return false; window.location.hash = `#/question/${q.next_id}`; },
  });

  const position = useMemo(() => {
    if (!q || !all.data) return null;
    const inChapter = all.data.items.filter((e) => e.chapter_id === q.chapter_id);
    const at = inChapter.findIndex((e) => e.id === q.id);
    return at < 0 ? null : { at: at + 1, of: inChapter.length };
  }, [q, all.data]);

  if (error) return <main><a href="#/">← All questions</a><ErrorNotice what={`question ${pad(id)}`} error={error} onRetry={retry} /></main>;
  if (!q || q.id !== id) return <main><a href="#/">← All questions</a><Skeleton variant="page" label={`Loading question ${pad(id)}`} /></main>;

  const done = prog.done[q.id] !== undefined;
  const marked = prog.bookmarks[q.id] !== undefined;
  const guide = q.supplemental ? "answer guide" : "solution";

  return (
    <main className="question">
      <Pager q={q} position={position} />
      <header className="qhead">
        <h1><span className="id">{pad(q.id)}</span> {q.title}</h1>
        <div className="qactions">
          <button type="button" className="toggle" aria-pressed={done} onClick={() => progress.toggleDone(q.id)}>{done ? "✓ Done" : "Mark done"}</button>
          <button type="button" className="toggle" aria-pressed={marked} onClick={() => progress.toggleBookmark(q.id)}>{marked ? "★ Bookmarked" : "☆ Bookmark"}</button>
        </div>
        <p className="meta">
          {q.chapter_id} · {q.chapter_title} · {q.subsection_title} · {q.language}
          {q.difficulty && <> · <span className="diff">{q.difficulty}</span></>}
          {q.lesson !== "none" && <> · <span className={`badge ${q.lesson}`}>{LESSON_LABEL[q.lesson]}</span></>}
        </p>
      </header>

      {q.articles.length > 0 && (
        <section aria-labelledby="learn-h" className="learn">
          <h2 id="learn-h">Learn this</h2>
          <ul className="lesson-cards">
            {q.articles.map((a) => (
              <li key={a.id} className="lesson-card">
                <span className="eyebrow">{PURPOSE_LABEL[a.purpose] || a.purpose}{a.review_state !== "complete" && ` · ${STATE_LABEL[a.review_state]}`}</span>
                <a href={`#/article/${a.id}`}>{a.title}</a>
                <span className="go" aria-hidden="true">Open the lesson →</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      <section aria-labelledby="task-h">
        <h2 id="task-h">Original task</h2>
        {q.source_url && <p className="sub"><a className="cta" href={q.source_url} target="_blank" rel="noopener noreferrer">Open this problem on {SOURCE_LABEL[q.source_key] || q.source_key} ↗</a></p>}
        <div className="task"><MarkdownPrompt text={q.prompt} /></div>
      </section>

      <section aria-labelledby="sol-h">
        <h2 id="sol-h">{q.supplemental ? "Answer guide" : "Solution"}</h2>
        <p className="sub">Try it first, then check your work.</p>
        <button type="button" aria-expanded={shown} aria-controls="solution-panel" onClick={() => setShown((v) => !v)}>
          {shown ? `Hide ${guide}` : `Show ${guide}`}
        </button>
        <div id="solution-panel" hidden={!shown}>
          {shown && !sol.data && !sol.error && <p role="status">Loading {guide}…</p>}
          {sol.error && <ErrorNotice what={`the ${guide}`} error={sol.error} onRetry={() => { setSol({ data: null, error: null }); setAttempt((n) => n + 1); }} />}
          {sol.data?.code && <CodeView code={sol.data.code} language={sol.data.language} label={`Solution code for question ${pad(id)}`} />}
          {sol.data?.answer && <div className="answer-guide"><MarkdownPrompt text={sol.data.answer} /></div>}
        </div>
      </section>

      <section aria-labelledby="reading-h" className="readings">
        <h2 id="reading-h">Best readings (easy to follow)</h2>
        <p className="sub">Two ranked references selected for technical depth, equations, implementation detail, or primary-source authority.</p>
        <ol>
          {q.resources.map((r) => (
            <li key={r.rank}>
              <a href={r.url} target="_blank" rel="noreferrer"><strong>{r.title}</strong></a>
              <span className="reading-meta">{r.publisher} · {r.access_note}</span>
              <p>{r.why}</p>
            </li>
          ))}
        </ol>
      </section>

      <footer className="source">
        <p>
          <SourceLink q={q} />
          {q.repository_url && <> · <a href={q.repository_url} target="_blank" rel="noreferrer">Repository</a></>}
          {q.pdf_page_first && <> · Solutions PDF, pages {q.pdf_page_first === q.pdf_page_last ? q.pdf_page_first : `${q.pdf_page_first}–${q.pdf_page_last}`}</>}
        </p>
        {q.license_note && <p className="license">{q.license_note}</p>}
      </footer>
    </main>
  );
}
