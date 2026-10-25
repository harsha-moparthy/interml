import { useMemo, useRef, useState } from "react";
import { useAllQuestions, useFetch, pad } from "./lib.js";
import { progress, downloadExport } from "./progress.js";
import { useProgress } from "./useProgress.js";
import { Bar, ErrorNotice, Skeleton } from "./ui.jsx";

const ago = (t) => {
  const m = Math.round((Date.now() - t) / 60000);
  if (m < 1) return "just now";
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  return h < 48 ? `${h} h ago` : `${Math.round(h / 24)} days ago`;
};

export function ProgressPage() {
  const prog = useProgress();
  const all = useAllQuestions();
  const chapters = useFetch("/api/chapters");
  const articles = useFetch("/api/articles", { cache: true });
  const [confirming, setConfirming] = useState(false);
  const [message, setMessage] = useState(null);
  const file = useRef(null);

  const byId = useMemo(() => Object.fromEntries((all.data?.items || []).map((e) => [e.id, e])), [all.data]);
  const articleTitle = useMemo(() => Object.fromEntries((articles.data || []).map((a) => [a.id, a.title])), [articles.data]);
  const doneIds = Object.keys(prog.done).map(Number);
  const bookmarkIds = Object.keys(prog.bookmarks).map(Number).sort((a, b) => prog.bookmarks[b] - prog.bookmarks[a]);
  const total = all.data?.total ?? 0;

  const chapterRows = useMemo(() => {
    const rows = {};
    for (const e of all.data?.items || []) {
      const r = (rows[e.chapter_id] ||= { id: e.chapter_id, title: e.chapter_title, count: 0, done: 0, next: null, first: e.id });
      r.count++;
      if (prog.done[e.id] !== undefined) r.done++; else if (r.next === null) r.next = e.id;
    }
    return Object.values(rows);
  }, [all.data, prog.done]);

  const empty = doneIds.length === 0 && bookmarkIds.length === 0 && prog.recent.length === 0;

  const onImport = async (e) => {
    const f = e.target.files?.[0];
    e.target.value = "";
    if (!f) return;
    try {
      const counts = progress.importText(await f.text());
      setMessage({ ok: true, text: `Imported ${counts.done} done and ${counts.bookmarks} bookmarked questions.` });
    } catch (err) {
      setMessage({ ok: false, text: `${err.message} Your progress was not changed.` });
    }
  };

  if (all.error) return <main><ErrorNotice what="the question list" error={all.error} onRetry={all.retry} /></main>;
  if (!all.data) return <main><Skeleton variant="page" label="Loading progress" /></main>;

  return (
    <main className="progresspage">
      <h1>Progress</h1>

      {empty ? (
        <div className="empty">
          <h2>Nothing here yet</h2>
          <p>Mark questions done and bookmark the ones to revisit. They show up on this page.</p>
          <p><a className="cta" href="#/question/1">Start with question 001 →</a></p>
        </div>
      ) : (
        <>
          <section className="pg-summary" aria-label="Overall progress">
            <p className="counts"><strong>{doneIds.length}</strong> of <strong>{total}</strong> questions done · <strong>{bookmarkIds.length}</strong> bookmarked</p>
            <Bar value={doneIds.length} max={total} label="Questions done" />
          </section>

          <section aria-labelledby="pg-chapters">
            <h2 id="pg-chapters">By chapter</h2>
            <ul className="pg-chapters">
              {chapterRows.map((c) => (
                <li key={c.id}>
                  <a href={`#/question/${c.next ?? c.first}`}>
                    <span className="cid">{c.id}</span>
                    <span className="ctitle">{c.title}</span>
                    <span className="ccount">{c.done} of {c.count}{c.next === null ? " · all done" : ""}</span>
                    <Bar value={c.done} max={c.count} label={`${c.id} done`} />
                  </a>
                </li>
              ))}
            </ul>
          </section>

          <div className="pg-cols">
            <section aria-labelledby="pg-bookmarks">
              <h2 id="pg-bookmarks">Bookmarks</h2>
              {bookmarkIds.length === 0 ? <p className="sub">No bookmarks yet. Choose the ☆ on a question.</p> : (
                <ul className="pg-list">
                  {bookmarkIds.map((id) => (
                    <li key={id}>
                      <a href={`#/question/${id}`}><span className="rid">{pad(id)}</span> {byId[id]?.title ?? `Question ${pad(id)}`}</a>
                      <button type="button" className="linklike" aria-label={`Remove bookmark from question ${pad(id)}`} onClick={() => progress.toggleBookmark(id)}>Remove</button>
                    </li>
                  ))}
                </ul>
              )}
            </section>
            <section aria-labelledby="pg-recent">
              <h2 id="pg-recent">Recently opened</h2>
              {prog.recent.length === 0 ? <p className="sub">Nothing opened yet.</p> : (
                <ul className="pg-list">
                  {prog.recent.map((r) => r.kind === "question" ? (
                    <li key={`q${r.id}`}><a href={`#/question/${r.id}`}><span className="rid">{pad(r.id)}</span> {byId[r.id]?.title ?? `Question ${pad(r.id)}`}</a><span className="sub inline">{ago(r.t)}</span></li>
                  ) : (
                    <li key={`a${r.id}`}><a href={`#/article/${r.id}`}><span className="rid">Lesson</span> {articleTitle[r.id] ?? r.id}</a><span className="sub inline">{ago(r.t)}</span></li>
                  ))}
                </ul>
              )}
            </section>
          </div>
        </>
      )}

      <section aria-labelledby="pg-data" className="pg-data">
        <h2 id="pg-data">Your data</h2>
        <p className="sub">Progress stays in this browser. Export a file to keep a copy or move it to another browser.</p>
        <div className="pg-actions">
          <button type="button" onClick={() => downloadExport(progress)}>Export progress</button>
          <button type="button" onClick={() => file.current?.click()}>Import progress…</button>
          <input ref={file} type="file" accept="application/json,.json" onChange={onImport} hidden aria-label="Progress file to import" />
          {!confirming && <button type="button" className="danger" onClick={() => setConfirming(true)} disabled={empty}>Reset all progress…</button>}
        </div>
        {confirming && (
          <div role="alertdialog" aria-labelledby="reset-q" className="pg-confirm">
            <p id="reset-q"><strong>Delete all progress?</strong> This removes {doneIds.length} done and {bookmarkIds.length} bookmarked questions and your reading positions. You can undo this only by importing an export file.</p>
            <button type="button" className="danger" onClick={() => { progress.reset(); setConfirming(false); setMessage({ ok: true, text: "All progress was deleted." }); }}>Delete everything</button>
            <button type="button" onClick={() => setConfirming(false)}>Cancel</button>
          </div>
        )}
        {message && <p role="status" className={message.ok ? "pg-ok" : "error"}>{message.text}</p>}
      </section>
    </main>
  );
}
