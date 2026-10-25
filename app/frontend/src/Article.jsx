import { useEffect, useMemo, useRef, useState } from "react";
import { useFetch, pad } from "./lib.js";
import { asset } from "./staticApi.js";
import { ErrorNotice, Skeleton } from "./ui.jsx";
import { progress } from "./progress.js";
import { useProgress } from "./useProgress.js";
import { Inline, Tex } from "./Inline.jsx";
import { CheckBlock } from "./demo/Check.jsx";
import { PlaygroundBlock } from "./py/Playground.jsx";
import { CodeView } from "./code/CodeView.jsx";
import { WalkthroughBlock } from "./code/Walkthrough.jsx";

const STATE_LABEL = { draft: "Draft", reviewed: "Reviewed", complete: "Complete" };
const CALLOUT_TITLE = { definition: "Definition", note: "Note", mistake: "Common mistake", boundary: "Boundary case", example: "Example" };


const CALLOUT_ICON = { definition: "◆", note: "ℹ", mistake: "⚠", boundary: "◇", example: "▶" };
const prefersReduced = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Text of an inline-markup string without the markup: used for ids and word counts. */
const plain = (t) => String(t || "").replace(/\[\[(?:article|q):[^\]|]+\|([^\]]*)\]\]/g, "$1").replace(/\$[^$]+\$/g, " ").replace(/[`*]/g, "");

function slugify(text) {
  return plain(text).toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60) || "section";
}

/** Unique ids for every H2 and H3, by block index, plus the H2 entries for the table of contents. */
function sectionIds(blocks) {
  const used = new Set();
  const byIndex = {};
  const toc = [];
  blocks.forEach((b, i) => {
    if (b.type !== "heading" || (b.level !== 2 && b.level !== 3)) return;
    let id = slugify(b.text), n = 2;
    while (used.has(id)) id = `${slugify(b.text)}-${n++}`;
    used.add(id);
    byIndex[i] = id;
    if (b.level === 2) toc.push({ id, text: b.text });
  });
  return { byIndex, toc };
}

function readingMinutes(blocks) {
  const words = blocks.reduce((n, b) => {
    const text = b.type === "list" ? b.items.join(" ") : [b.text, b.caption, b.title, b.headline, b.subtitle].filter(Boolean).join(" ");
    return n + plain(text).split(/\s+/).filter(Boolean).length;
  }, 0);
  return Math.max(1, Math.round(words / 200));
}

/**
 * The section being read: the last one whose heading has passed the reading line near the top of the screen.
 * At the very bottom of the page the last visible heading counts, because short final sections cannot reach the top.
 */
function useCurrentSection(ids) {
  const [current, setCurrent] = useState(null);
  const key = ids.join("|");
  useEffect(() => {
    if (!ids.length) return undefined;
    let frame = 0;
    const measure = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const atBottom = window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2;
        const line = atBottom ? window.innerHeight * 0.9 : 120;
        let now = null;
        for (const id of ids) {
          const el = document.getElementById(id);
          if (el && el.getBoundingClientRect().top <= line) now = id;
        }
        setCurrent(now ?? ids[0]);
      });
    };
    measure();
    window.addEventListener("scroll", measure, { passive: true });
    window.addEventListener("resize", measure);
    return () => { cancelAnimationFrame(frame); window.removeEventListener("scroll", measure); window.removeEventListener("resize", measure); };
  }, [key]);   // eslint-disable-line react-hooks/exhaustive-deps
  return current;
}

/** Fraction of the page scrolled (0..1); saves it for the article once the learner has really started reading. */
function useReadingProgress(articleId, enabled) {
  const [fraction, setFraction] = useState(0);
  useEffect(() => {
    if (!enabled) return undefined;
    let frame = 0;
    const on = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const room = document.documentElement.scrollHeight - window.innerHeight;
        const f = room > 0 ? Math.min(1, Math.max(0, window.scrollY / room)) : 0;
        setFraction(f);
        if (window.scrollY > 200) progress.setReading(articleId, f);
      });
    };
    on();
    window.addEventListener("scroll", on, { passive: true });
    return () => { cancelAnimationFrame(frame); window.removeEventListener("scroll", on); };
  }, [articleId, enabled]);
  return fraction;
}

function Block({ b, id, media, demos, checks, playgrounds }) {
  switch (b.type) {
    case "heading": {
      const H = `h${b.level}`;
      return <H id={id} tabIndex={id ? -1 : undefined}><Inline text={b.text} /></H>;
    }
    case "paragraph": return <p><Inline text={b.text} /></p>;
    case "list": {
      const L = b.ordered ? "ol" : "ul";
      return <L>{b.items.map((t, i) => <li key={i}><Inline text={t} /></li>)}</L>;
    }
    case "equation":
      return (
        <figure className="equation">
          <div className="eq" role="math" aria-label={b.latex}><Tex tex={b.latex} display /></div>
          {b.caption && <figcaption><Inline text={b.caption} /></figcaption>}
        </figure>
      );
    case "code":
      return (
        <figure>
          <CodeView code={b.code} language={b.language === "cpp" ? "C++" : "Python"} label={b.caption || "Code example"} maxHeight="26rem" />
          {b.caption && <figcaption><Inline text={b.caption} /></figcaption>}
        </figure>
      );
    case "image": {
      const m = media[b.path] || {};
      return (
        <figure>
          <img src={asset(`media/${b.path}`)} alt={m.alt_text || ""} />
          {(b.caption || m.attribution) && (
            <figcaption>{b.caption && <Inline text={b.caption} />}{m.attribution && <span className="sub"> {m.attribution}</span>}</figcaption>
          )}
        </figure>
      );
    }
    case "callout":
      return (
        <aside className={`callout ${b.kind}`}>
          <strong><span className="ico" aria-hidden="true">{CALLOUT_ICON[b.kind]}</span>{b.title || CALLOUT_TITLE[b.kind]}</strong>
          <p><Inline text={b.text} /></p>
        </aside>
      );
    case "table":
      return (
        <table className="plain">
          {b.caption && <caption>{b.caption}</caption>}
          <thead><tr>{b.header.map((h, i) => <th key={i} scope="col"><Inline text={h} /></th>)}</tr></thead>
          <tbody>{b.rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}><Inline text={c} /></td>)}</tr>)}</tbody>
        </table>
      );
    case "demonstration":
      return null;
    case "check":
      return <CheckBlock id={b.id} check={checks[b.id]} />;
    case "compare":
      return null;
    case "walkthrough":
      return <WalkthroughBlock exercise={b.exercise_id} title={b.title} notes={b.notes} />;
    case "hero":
      return null;   // rendered in the page header
    case "gallery":
      return (
        <div className="gallery">
          {b.items.map((it) => (
            <figure key={it.path}><img src={asset(`media/${it.path}`)} alt={(media[it.path] || {}).alt_text || ""} loading="lazy" /><figcaption><Inline text={it.caption} /></figcaption></figure>
          ))}
        </div>
      );
    case "playground":
      return <PlaygroundBlock exercise={b.exercise_id} pg={playgrounds[b.exercise_id]} />;
    default: return null;
  }
}

export function ArticlePage({ id }) {
  const { data: a, error, retry } = useFetch(`/api/articles/${id}`);
  const prog = useProgress();
  const [tocOpen, setTocOpen] = useState(false);
  const [resume, setResume] = useState(null);
  const loaded = !!a && a.id === id;
  const saved = useRef(null);

  const sections = useMemo(() => (loaded ? sectionIds(a.blocks) : { byIndex: {}, toc: [] }), [a, loaded]);
  const minutes = useMemo(() => (loaded ? readingMinutes(a.blocks) : 0), [a, loaded]);
  const current = useCurrentSection(sections.toc.map((t) => t.id));
  const fraction = useReadingProgress(id, loaded);

  useEffect(() => { window.scrollTo(0, 0); setTocOpen(false); setResume(null); saved.current = null; }, [id]);
  useEffect(() => {
    if (!loaded) return;
    progress.touch("article", id);
    const r = progress.getSnapshot().reading[id];
    saved.current = r || null;
    setResume(r && r.y > 0.03 && r.y < 0.97 ? r.y : null);
  }, [loaded, id]);

  const back = <nav className="pager"><a href="#/">← All questions</a></nav>;
  if (error) return <main>{back}<ErrorNotice what={`article “${id}”`} error={error} onRetry={retry} /></main>;
  if (!loaded) return <main>{back}<Skeleton variant="article" label="Loading article" /></main>;

  const goTo = (e, target) => {
    e.preventDefault();
    const el = document.getElementById(target);
    if (!el) return;
    el.scrollIntoView({ behavior: prefersReduced() ? "auto" : "smooth", block: "start" });
    el.focus({ preventScroll: true });
    setTocOpen(false);
  };
  const doResume = () => {
    const room = document.documentElement.scrollHeight - window.innerHeight;
    window.scrollTo({ top: resume * room, behavior: prefersReduced() ? "auto" : "smooth" });
    setResume(null);
  };
  const hero = a.blocks[0]?.type === "hero" ? a.blocks[0] : null;
  const kind = a.kind === "lesson" ? "Deep lesson" : "Prerequisite article";
  const showToc = sections.toc.length > 2;
  const pct = Math.round(fraction * 100);
  void prog;

  return (
    <main className="article">
      <div className="readbar" role="progressbar" aria-label="Reading progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={pct}><span style={{ width: `${pct}%` }} /></div>
      <nav className="pager" aria-label="Article navigation">
        <a href="#/">← All questions</a>
        {a.exercises.map((e) => <a key={e.id} href={`#/question/${e.id}`}>Question {pad(e.id)}</a>)}
      </nav>
      <header className="arthead">
        <p className="eyebrow">{kind}{hero ? ` · ${a.title}` : ""}</p>
        <h1>{hero ? hero.headline : a.title}</h1>
        {hero?.subtitle && <p className="herosub">{hero.subtitle}</p>}
        <p className="meta">
          {kind} · <span className={`badge ${a.review_state === "complete" ? "complete" : "planned"}`}>{STATE_LABEL[a.review_state]}</span>
          {" "}· <span className="readtime">{minutes} min read</span> · revision {a.revision}
        </p>
      </header>
      {a.prerequisites.length > 0 && (
        <section aria-labelledby="prereq-h" className="prereq">
          <h2 id="prereq-h">Before you start</h2>
          <ul>
            {a.prerequisites.map((p) => (
              <li key={p.id}><a href={`#/article/${p.id}`}>{p.title}</a>{p.review_state !== "complete" && <span className="sub inline"> ({STATE_LABEL[p.review_state]})</span>}</li>
            ))}
          </ul>
        </section>
      )}
      {resume !== null && (
        <p className="resume" role="status">
          Pick up where you left off, about {Math.round(resume * 100)}% of the way through.
          <button type="button" onClick={doResume}>Continue</button>
          <button type="button" className="linklike" onClick={() => setResume(null)}>Start from the top</button>
        </p>
      )}
      <div className={`article-layout${showToc ? " has-toc" : ""}`}>
        {showToc && (
          <nav aria-label="On this page" className="toc" data-open={tocOpen}>
            <button type="button" className="toc-toggle" aria-expanded={tocOpen} aria-controls="toc-list" onClick={() => setTocOpen((v) => !v)}>On this page</button>
            <strong className="toc-title">On this page</strong>
            <ol id="toc-list" className="toc-list">
              {sections.toc.map((t) => (
                <li key={t.id}>
                  <a href={window.location.hash} onClick={(e) => goTo(e, t.id)} aria-current={current === t.id ? "location" : undefined}><Inline text={t.text} /></a>
                </li>
              ))}
            </ol>
          </nav>
        )}
        <div className="article-body">
          {a.blocks.map((b, i) => <Block key={i} b={b} id={sections.byIndex[i]} media={a.media} demos={a.demonstrations} checks={a.checks} playgrounds={a.playgrounds} />)}
        </div>
      </div>
      {a.required_by.length > 0 && (
        <section><h2>Used by</h2>
          <ul>{a.required_by.map((r) => <li key={r.id}><a href={`#/article/${r.id}`}>{r.title}</a></li>)}</ul>
        </section>
      )}
      {a.sources.length > 0 && (
        <footer className="source"><h2>Sources</h2>
          <ul>{a.sources.map((s, i) => <li key={i}>{s.url ? <a href={s.url} target="_blank" rel="noreferrer">{s.title}</a> : s.title}</li>)}</ul>
        </footer>
      )}
    </main>
  );
}
