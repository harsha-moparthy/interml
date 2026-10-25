import { useCallback, useEffect, useRef, useState } from "react";
import { Compartment, EditorState, StateEffect, StateField } from "@codemirror/state";
import { Decoration, EditorView, drawSelection, keymap, lineNumbers } from "@codemirror/view";
import { bracketMatching, indentOnInput, indentUnit, syntaxHighlighting } from "@codemirror/language";
import { defaultKeymap, history, historyKeymap, indentWithTab } from "@codemirror/commands";
import { highlightSelectionMatches, searchKeymap } from "@codemirror/search";
import { python } from "@codemirror/lang-python";
import { cpp } from "@codemirror/lang-cpp";
import { tagHighlighter, tags as t } from "@lezer/highlight";

// Token classes map to CSS variables in styles.css, so light and dark themes need no JavaScript.
const highlighter = tagHighlighter([
  { tag: [t.keyword, t.controlKeyword, t.moduleKeyword, t.definitionKeyword, t.operatorKeyword], class: "tok-keyword" },
  { tag: [t.string, t.special(t.string), t.regexp], class: "tok-string" },
  { tag: [t.number, t.integer, t.float], class: "tok-number" },
  { tag: [t.bool, t.null, t.self, t.atom], class: "tok-atom" },
  { tag: [t.comment, t.lineComment, t.blockComment, t.docComment], class: "tok-comment" },
  { tag: [t.function(t.variableName), t.function(t.propertyName), t.macroName], class: "tok-function" },
  { tag: [t.definition(t.variableName), t.definition(t.function(t.variableName))], class: "tok-def" },
  { tag: [t.className, t.typeName, t.namespace], class: "tok-type" },
  { tag: [t.propertyName, t.attributeName], class: "tok-property" },
  { tag: [t.operator, t.punctuation, t.bracket], class: "tok-punct" },
  { tag: [t.meta, t.annotation], class: "tok-meta" },
]);

const theme = EditorView.theme({
  "&": { backgroundColor: "var(--code-bg)", color: "var(--code-fg)", fontSize: "0.88rem" },
  ".cm-content": { fontFamily: "ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, Consolas, monospace", padding: "8px 0", caretColor: "var(--code-fg)" },
  ".cm-gutters": { backgroundColor: "var(--code-gutter-bg)", color: "var(--code-gutter-fg)", border: "none", borderRight: "1px solid var(--line)" },
  ".cm-lineNumbers .cm-gutterElement": { padding: "0 10px 0 12px", minWidth: "2.2em" },
  ".cm-line": { padding: "0 14px 0 10px" },
  "&.cm-focused": { outline: "3px solid var(--accent)", outlineOffset: "1px" },
  ".cm-selectionBackground, &.cm-focused .cm-selectionBackground": { backgroundColor: "var(--code-selection) !important" },
  ".cm-scroller": { overflow: "auto", lineHeight: "1.55" },
});

// ---- line highlights: [{from, to, id, kind}] with 1-based inclusive lines
const setHighlights = StateEffect.define();
const highlightField = StateField.define({
  create: () => Decoration.none,
  update(deco, tr) {
    for (const e of tr.effects) {
      if (e.is(setHighlights)) {
        const { ranges, active, doc } = e.value;
        const marks = [];
        for (const r of ranges) {
          for (let n = Math.max(r.from, 1); n <= Math.min(r.to, doc.lines); n++) {
            const cls = `cm-hl cm-hl-${r.kind || "note"}${r.id === active ? " cm-hl-active" : ""}`;
            marks.push(Decoration.line({ class: cls, attributes: { "data-hl": String(r.id) } }).range(doc.line(n).from));
          }
        }
        marks.sort((a, b) => a.from - b.from);
        return Decoration.set(marks, true);
      }
    }
    return deco.map(tr.changes);
  },
  provide: (f) => EditorView.decorations.from(f),
});

const languageFor = (name) => (/c\+\+|cpp|cuda/i.test(name || "") ? cpp() : python());

function useClipboard() {
  const [copied, setCopied] = useState(false);
  const copy = useCallback(async (text) => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text; document.body.appendChild(ta); ta.select();
      try { document.execCommand("copy"); } finally { ta.remove(); }
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  }, []);
  return [copied, copy];
}

/** Read-only code viewer with line numbers, syntax colours, optional line-range highlights, wrap toggle, and copy. */
export function CodeView({ code, language = "Python", highlights = [], activeId = null, onHighlightClick, label = "Code",
                           maxHeight = "34rem", scrollToActive = true }) {
  const host = useRef(null);
  const view = useRef(null);
  const wrapComp = useRef(new Compartment());
  const [wrap, setWrap] = useState(false);
  const [copied, copy] = useClipboard();
  const clickRef = useRef(onHighlightClick);
  clickRef.current = onHighlightClick;
  const lines = code.replace(/\n$/, "").split("\n").length;

  useEffect(() => {
    const state = EditorState.create({
      doc: code,
      extensions: [
        lineNumbers(), languageFor(language), syntaxHighlighting(highlighter), theme, highlightField, drawSelection(),
        highlightSelectionMatches(), keymap.of(searchKeymap), wrapComp.current.of([]),
        EditorState.readOnly.of(true), EditorView.editable.of(false),
        EditorView.contentAttributes.of({ "aria-label": label, tabindex: "0", role: "textbox", "aria-readonly": "true" }),
        EditorView.domEventHandlers({
          click(e, v) {
            const el = e.target.closest?.("[data-hl]");
            if (el && clickRef.current) clickRef.current(el.getAttribute("data-hl"));
          },
        }),
      ],
    });
    view.current = new EditorView({ state, parent: host.current });
    return () => view.current.destroy();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const v = view.current;
    if (v.state.doc.toString() !== code) v.dispatch({ changes: { from: 0, to: v.state.doc.length, insert: code } });
  }, [code]);

  useEffect(() => {
    const v = view.current;
    v.dispatch({ effects: setHighlights.of({ ranges: highlights, active: activeId, doc: v.state.doc }) });
    if (scrollToActive && activeId != null) {
      const r = highlights.find((h) => String(h.id) === String(activeId));
      if (r && r.from <= v.state.doc.lines) {
        v.dispatch({ effects: EditorView.scrollIntoView(v.state.doc.line(r.from).from, { y: "center" }) });
      }
    }
  }, [highlights, activeId, code, scrollToActive]);

  useEffect(() => {
    view.current.dispatch({ effects: wrapComp.current.reconfigure(wrap ? EditorView.lineWrapping : []) });
  }, [wrap]);

  return (
    <figure className="codeview-box">
      <div className="codebar">
        <span className="lang">{language}</span>
        <span className="count">{lines} line{lines === 1 ? "" : "s"}</span>
        <span className="spacer" />
        <button type="button" onClick={() => setWrap((w) => !w)} aria-pressed={wrap}>Wrap lines</button>
        <button type="button" onClick={() => copy(code)}>{copied ? "Copied" : "Copy"}</button>
        <span role="status" aria-live="polite" className="sr-only">{copied ? "Code copied to the clipboard" : ""}</span>
      </div>
      <div ref={host} className="cmhost" style={{ maxHeight }} />
    </figure>
  );
}

/** Editable code area for the playground. Ctrl/Cmd+Enter runs. Press Esc then Tab to leave the editor with the keyboard. */
export function CodeEditor({ value, onChange, onRun, label = "Python code", language = "Python" }) {
  const host = useRef(null);
  const view = useRef(null);
  const cb = useRef({ onChange, onRun });
  cb.current = { onChange, onRun };

  useEffect(() => {
    const run = () => { cb.current.onRun?.(); return true; };
    const state = EditorState.create({
      doc: value,
      extensions: [
        lineNumbers(), languageFor(language), syntaxHighlighting(highlighter), theme, history(), drawSelection(),
        indentOnInput(), bracketMatching(), indentUnit.of("    "), highlightSelectionMatches(),
        keymap.of([{ key: "Mod-Enter", run }, indentWithTab, ...defaultKeymap, ...historyKeymap, ...searchKeymap]),
        EditorView.lineWrapping,
        EditorView.contentAttributes.of({ "aria-label": label, "aria-multiline": "true" }),
        EditorView.updateListener.of((u) => { if (u.docChanged) cb.current.onChange(u.state.doc.toString()); }),
      ],
    });
    view.current = new EditorView({ state, parent: host.current });
    return () => view.current.destroy();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const v = view.current;
    if (v.state.doc.toString() !== value) v.dispatch({ changes: { from: 0, to: v.state.doc.length, insert: value } });
  }, [value]);

  return <div ref={host} className="cmhost editor" />;
}
