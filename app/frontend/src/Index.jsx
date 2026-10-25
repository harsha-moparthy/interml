import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useFetch, useAllQuestions, pad } from "./lib.js";
import { STATIC } from "./staticApi.js";
import { useProgress } from "./useProgress.js";
import { progress } from "./progress.js";
import { Bar, EmptyState, ErrorNotice, Skeleton } from "./ui.jsx";
import { onSearchRequest, takeSearchRequest } from "./shortcuts.js";

const LESSON_LABEL = { complete: "Deep lesson available", planned: "Deep lesson planned", none: "" };
const WHY_LABEL = { id: "ID match", keyword: "Keywords", meaning: "Meaning" };
const STATES = [["all", "All"], ["todo", "Not done"], ["done", "Done"], ["bookmarked", "Bookmarked"]];
const EXAMPLES = [
  ["kl divergence", "Abbreviations work"],
  ["make training use less memory", "Describe the idea, not the title"],
  ["\"causal mask\"", "Exact phrase"],
  ["attention -dropout", "Leave a word out"],
  ["chapter:attention softmax", "Search inside one chapter"],
  ["id:300-330", "A range of question IDs"],
  ["lesson:planned", "Questions with a planned deep lesson"],
  ["lang:cpp", "C++ and CUDA questions"],
  ["difficulty:hard pandas", "Filter by difficulty"],
];

function useDebounced(value, ms) {
  const [v, setV] = useState(value);
  useEffect(() => { const t = setTimeout(() => setV(value), ms); return () => clearTimeout(t); }, [value, ms]);
  return v;
}

function Marked({ parts }) {
  return parts.map((p, i) => (p.hit ? <mark key={i}>{p.text}</mark> : <span key={i}>{p.text}</span>));
}

// The index keeps its scroll position while the learner reads a question, then restores it on return.
let savedScroll = 0;

function Summary({ stats, done }) {
  const { recent } = useProgress();
  const last = recent[0];
  const total = stats?.imported_questions ?? 0;
  const resume = last
    ? { href: last.kind === "question" ? `#/question/${last.id}` : `#/article/${last.id}`, label: last.kind === "question" ? `Resume question ${pad(last.id)}` : "Resume your lesson" }
    : { href: "#/question/1", label: "Start with question 001" };
  return (
    <section className="summary" aria-label="Your progress">
      <div>
        <h1>ML coding questions</h1>
        {stats ? (
          <p className="counts">
            <strong>{done}</strong> of <strong>{total}</strong> questions done ·{" "}
            <strong>{stats.completed_deep_lessons}</strong> of {stats.planned_deep_lessons} deep lessons available
          </p>
        ) : <p className="counts">Loading…</p>}
        <Bar value={done} max={total} label="Questions done" />
      </div>
      <a className="cta" href={resume.href}>{resume.label} →</a>
    </section>
  );
}

function ChapterNav({ chapters, chapter, onPick, doneByChapter }) {
  return (
    <nav className="chapters" aria-label="Chapters">
      <h2>Chapters</h2>
      <ul>
        <li><button type="button" aria-current={chapter === "" ? "true" : undefined} onClick={() => onPick("")}><span className="ctitle">All chapters</span></button></li>
        {chapters.map((c) => {
          const d = doneByChapter[c.id] || 0;
          return (
            <li key={c.id}>
              <button type="button" aria-current={chapter === c.id ? "true" : undefined} onClick={() => onPick(c.id)}>
                <span className="cid">{c.id}</span>
                <span className="ctitle">{c.title}</span>
                <span className="ccount">{d} of {c.count}</span>
                <Bar value={d} max={c.count} label={`${c.id} done`} />
              </button>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}

function Card({ e, searching, done, marked }) {
  return (
    <article className={`result${done ? " is-done" : ""}`}>
      <div className="result-row">
        <a className="result-link" href={`#/question/${e.id}`}>
          <span className="rid">{pad(e.id)}</span>
          <span className="rtitle">{e.title_parts ? <Marked parts={e.title_parts} /> : e.title}</span>
        </a>
        <span className="result-actions">
          <button type="button" className="mark" aria-pressed={done} aria-label={`Mark question ${pad(e.id)} done`} title="Mark done" onClick={() => progress.toggleDone(e.id)}>{done ? "✓" : "○"}</button>
          <button type="button" className="mark" aria-pressed={marked} aria-label={`Bookmark question ${pad(e.id)}`} title="Bookmark" onClick={() => progress.toggleBookmark(e.id)}>{marked ? "★" : "☆"}</button>
        </span>
      </div>
      {searching && e.snippet_parts && <p className="snippet"><Marked parts={e.snippet_parts} /></p>}
      <p className="rmeta">
        <span>{e.chapter_id} · {e.chapter_title}</span>
        {e.difficulty && <span className={`diff ${e.difficulty}`}>{e.difficulty}</span>}
        <span>{e.language}</span>
        {(e.why || []).map((w) => <span key={w} className={`why ${w}`}>{WHY_LABEL[w]}</span>)}
        {e.lesson !== "none" && <span className={`badge ${e.lesson}`}>{LESSON_LABEL[e.lesson]}</span>}
        {done && <span className="state done">✓ Done</span>}
        {marked && <span className="state bookmarked">★ Bookmarked</span>}
      </p>
    </article>
  );
}

export function IndexPage({ filters, setFilters }) {
  const { query, chapter, lesson, state = "all" } = filters;
  const set = (patch) => setFilters((f) => ({ ...f, ...patch }));
  const q = useDebounced(query, 180);
  const [limit, setLimit] = useState(40);
  const [order, setOrder] = useState("relevance");
  const input = useRef(null);
  const list = useRef(null);
  const stats = useFetch("/api/stats");
  const chapters = useFetch("/api/chapters");
  const all = useAllQuestions();
  const prog = useProgress();
  const searching = q.trim().length > 0;
  const byState = state !== "all";
  // A learner-state filter works on the whole result set, so ask for everything and filter here.
  const url = useMemo(() => `/api/search?${new URLSearchParams({ q, chapter, lesson, limit: searching && !byState ? limit : 1100 })}`, [q, chapter, lesson, limit, searching, byState]);
  const res = useFetch(url);
  useEffect(() => setLimit(40), [q, chapter, lesson, state]);

  const doneByChapter = useMemo(() => {
    const m = {};
    for (const e of all.data?.items || []) if (prog.done[e.id] !== undefined) m[e.chapter_id] = (m[e.chapter_id] || 0) + 1;
    return m;
  }, [all.data, prog.done]);

  // The header and the "/" key ask for the search box; take a request made before this page appeared, and any later one.
  useEffect(() => {
    const take = () => { if (takeSearchRequest()) input.current?.focus(); };
    take();
    return onSearchRequest(take);
  }, []);

  const move = (e, dir) => {
    const links = [...list.current.querySelectorAll("a.result-link")];
    const at = links.indexOf(document.activeElement);
    const next = links[Math.min(Math.max(at + dir, 0), links.length - 1)];
    if (next) { e.preventDefault(); next.focus(); }
  };
  const onInputKey = (e) => {
    if (e.key === "ArrowDown") move(e, 1);
    if (e.key === "Escape") set({ query: "" });
  };
  const onListKey = (e) => {
    if (e.key === "ArrowDown") move(e, 1);
    if (e.key === "ArrowUp") {
      const links = [...list.current.querySelectorAll("a.result-link")];
      if (links[0] === document.activeElement) { e.preventDefault(); input.current?.focus(); } else move(e, -1);
    }
  };

  const data = res.data;
  const matching = useMemo(() => {
    if (!data) return [];
    let rows = !searching || order === "relevance" ? data.items : [...data.items].sort((a, b) => a.id - b.id);
    if (state === "todo") rows = rows.filter((e) => prog.done[e.id] === undefined);
    if (state === "done") rows = rows.filter((e) => prog.done[e.id] !== undefined);
    if (state === "bookmarked") rows = rows.filter((e) => prog.bookmarks[e.id] !== undefined);
    return rows;
  }, [data, searching, order, state, prog.done, prog.bookmarks]);
  const items = searching && byState ? matching.slice(0, limit) : matching;
  const total = byState ? matching.length : data?.total ?? 0;
  const anyFilter = query || chapter || lesson || byState;
  const doneTotal = Object.keys(prog.done).length;
  let lastChapter = null;

  // Restore the scroll position once the list is on screen after returning from a question.
  const restored = useRef(false);
  useLayoutEffect(() => {
    if (!restored.current && data && savedScroll > 0) { window.scrollTo(0, savedScroll); }
    if (data) restored.current = true;
  }, [data]);
  // Track scrolling while mounted. The listener must go in a layout effect: its cleanup runs inside the commit that removes
  // this page, before the browser can report the scroll jump caused by the page getting shorter.
  useLayoutEffect(() => {
    const on = () => { if (restored.current) savedScroll = window.scrollY; };   // not while the page is still a short skeleton
    window.addEventListener("scroll", on, { passive: true });
    return () => window.removeEventListener("scroll", on);
  }, []);

  const emptyMessage = state === "bookmarked" && !anyFilterBesidesState(filters)
    ? <EmptyState title="No bookmarks yet">Choose the ☆ on a question to keep it here.</EmptyState>
    : <EmptyState title="No questions match">Try fewer words, a different phrase, or clear the filters.</EmptyState>;

  return (
    <main className="browse">
      <Summary stats={stats.data} done={doneTotal} />

      <div className="browse-grid">
        {chapters.data && <ChapterNav chapters={chapters.data} chapter={chapter} onPick={(c) => set({ chapter: c })} doneByChapter={doneByChapter} />}

        <div className="browse-main">
          <form role="search" className="filters" onSubmit={(e) => e.preventDefault()}>
            <label className="grow">Search questions
              <input ref={input} type="search" value={query} onChange={(e) => set({ query: e.target.value })} onKeyDown={onInputKey}
                placeholder='Try: kl divergence · make training use less memory' autoComplete="off" spellCheck={false} />
            </label>
            <label className="chapter-select">Chapter
              <select value={chapter} onChange={(e) => set({ chapter: e.target.value })}>
                <option value="">All chapters</option>
                {(chapters.data || []).map((c) => <option key={c.id} value={c.id}>{c.id} · {c.title} ({c.count})</option>)}
              </select>
            </label>
            <label>Deep lesson
              <select value={lesson} onChange={(e) => set({ lesson: e.target.value })}>
                <option value="">Any</option>
                <option value="complete">Available</option>
                <option value="planned">Planned</option>
                <option value="none">No deep lesson</option>
              </select>
            </label>
            <button type="button" onClick={() => setFilters({ query: "", chapter: "", lesson: "", state: "all" })} disabled={!anyFilter}>Clear</button>
          </form>

          <div role="group" aria-label="Show questions" className="statefilter">
            {STATES.map(([v, label]) => (
              <button key={v} type="button" aria-pressed={state === v} onClick={() => set({ state: v })}>{label}</button>
            ))}
          </div>

          <details className="tips">
            <summary>Search tips</summary>
            <p>Search matches words, spelling mistakes, abbreviations, and meaning. Press <kbd>/</kbd> to jump to the search box, <kbd>↓</kbd> to move through results.</p>
            <ul>
              {EXAMPLES.map(([ex, what]) => (
                <li key={ex}><button type="button" className="linklike" onClick={() => { set({ query: ex }); input.current?.focus(); }}><code>{ex}</code></button> — {what}</li>
              ))}
            </ul>
          </details>

          {(res.error || chapters.error) && (
            <ErrorNotice what="the questions" error={res.error || chapters.error} onRetry={() => { res.error && res.retry(); chapters.error && chapters.retry(); }}>
              <p className="sub">Start the service and run the import, then try again.</p>
            </ErrorNotice>
          )}
          {!data && !res.error && <Skeleton variant="list" label="Loading questions" />}

          <div className="statusbar">
            <p role="status" aria-live="polite" className="result-count">
              {data ? (searching ? `${total} result${total === 1 ? "" : "s"} for “${q.trim()}”` : `${total} questions in original order`) : "Loading…"}
              {data && searching && <span className="sub inline"> · {data.semantic ? "keywords + meaning" : "keywords only"}</span>}
            </p>
            {searching && (
              <div role="group" aria-label="Sort order" className="sorter">
                <button type="button" aria-pressed={order === "relevance"} onClick={() => setOrder("relevance")}>Best match</button>
                <button type="button" aria-pressed={order === "id"} onClick={() => setOrder("id")}>Question order</button>
              </div>
            )}
          </div>
          {data && !data.semantic && searching && (
            <p className="sub">{STATIC ? "This online copy searches by keywords. Search by meaning needs the local model; run the site locally to use it." : <>Search by meaning is off because the local model is not installed. Run <code>python -m placement.semantic download</code>, then the import.</>}</p>
          )}
          {data && Object.keys(data.corrected || {}).length > 0 && (
            <p className="didyoumean">Showing results for {Object.entries(data.corrected).map(([from, to]) => (
              <span key={from}><strong>{to}</strong> (not “{from}”) </span>))}</p>
          )}
          {data && Object.entries(data.filters || {}).length > 0 && (
            <ul className="chips" aria-label="Active filters">
              {Object.entries(data.filters).flatMap(([k, vs]) => vs.map((v) => <li key={k + v}>{k}: {v}</li>))}
            </ul>
          )}

          <div ref={list} onKeyDown={onListKey}>
            {items.map((e) => {
              const head = !searching && e.chapter_id !== lastChapter ? (lastChapter = e.chapter_id, true) : false;
              return (
                <div key={e.id}>
                  {head && <h2 className="chapter-head">{e.chapter_id} · {e.chapter_title}</h2>}
                  <Card e={e} searching={searching} done={prog.done[e.id] !== undefined} marked={prog.bookmarks[e.id] !== undefined} />
                </div>
              );
            })}
          </div>
          {data && searching && total > items.length && (
            <p><button type="button" onClick={() => setLimit((l) => l + 40)}>Show 40 more ({total - items.length} left)</button></p>
          )}
          {data && total === 0 && emptyMessage}
        </div>
      </div>
    </main>
  );
}

function anyFilterBesidesState(f) { return !!(f.query || f.chapter || f.lesson); }
