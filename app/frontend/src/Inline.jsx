import { Fragment } from "react";
import katex from "katex";
import "katex/dist/katex.min.css";
import { pad } from "./lib.js";

export function Tex({ tex, display }) {
  let html;
  try {
    html = katex.renderToString(tex, { displayMode: display, throwOnError: false, trust: false, output: "htmlAndMathml" });
  } catch {
    html = null;
  }
  return html ? <span dangerouslySetInnerHTML={{ __html: html }} /> : <code>{tex}</code>;
}

// Inline markup: `code`, **bold**, *emphasis*, $math$, [[article:id|label]], [[q:316|label]].
const INLINE = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\$[^$]+\$|\[\[(?:article|q):[^\]]+\]\]|\[[^\]\n]+\]\(https?:\/\/[^\s)]+\))/g;

export function Inline({ text }) {
  return text.split(INLINE).map((part, i) => {
    if (!part) return null;
    if (part.startsWith("`")) return <code key={i}>{part.slice(1, -1)}</code>;
    if (part.startsWith("**")) return <strong key={i}><Inline text={part.slice(2, -2)} /></strong>;
    if (part.startsWith("*")) return <em key={i}><Inline text={part.slice(1, -1)} /></em>;
    if (part.startsWith("$")) return <Tex key={i} tex={part.slice(1, -1)} />;
    if (part.startsWith("[[")) {
      const [, kind, target, label] = part.match(/^\[\[(article|q):([^\]|]+)(?:\|([^\]]*))?\]\]$/);
      return kind === "q"
        ? <a key={i} href={`#/question/${target}`}>{label || `Question ${pad(Number(target))}`}</a>
        : <a key={i} href={`#/article/${target}`}>{label || target}</a>;
    }
    const link = part.match(/^\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)$/);
    if (link) {
      const [, label, href] = link;
      return <a key={i} href={href} target="_blank" rel="noopener noreferrer">{label}</a>;
    }
    return <Fragment key={i}>{part}</Fragment>;
  });
}

