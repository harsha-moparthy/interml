import { useEffect, useRef, useState } from "react";
import { THEMES, setTheme, useTheme } from "./theme.js";
import { useProgress } from "./useProgress.js";
import { requestSearchFocus, useShortcuts } from "./shortcuts.js";

const THEME_LABEL = { light: "Light", dark: "Dark", system: "System" };

/** Every shortcut the site offers. The help overlay lists exactly these. */
export const SHORTCUTS = [
  { keys: ["/"], what: "Go to search" },
  { keys: ["?"], what: "Show this list" },
  { keys: ["↓", "↑"], what: "Move through search results" },
  { keys: ["←", "→"], what: "Previous or next question" },
  { keys: ["Esc"], what: "Clear the search, or close this list" },
];

function goToSearch() {
  if (!/^#\/?$/.test(window.location.hash)) window.location.hash = "#/";
  requestSearchFocus();
}

function ThemeSwitch() {
  const theme = useTheme();
  return (
    <div role="group" aria-label="Colour theme" className="themeswitch">
      {THEMES.map((t) => (
        <button key={t} type="button" aria-pressed={theme === t} onClick={() => setTheme(t)}>{THEME_LABEL[t]}</button>
      ))}
    </div>
  );
}

function ShortcutHelp({ open, onClose }) {
  const ref = useRef(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog ref={ref} className="helpdialog" aria-labelledby="help-h" onClose={onClose}
      onClick={(e) => { if (e.target === ref.current) onClose(); }}>
      <h2 id="help-h">Keyboard shortcuts</h2>
      <dl>
        {SHORTCUTS.map((s) => (
          <div key={s.what}>
            <dt>{s.keys.map((k) => <kbd key={k}>{k}</kbd>)}</dt>
            <dd>{s.what}</dd>
          </div>
        ))}
      </dl>
      <p className="sub">Shortcuts do nothing while you type in a box.</p>
      <button type="button" onClick={onClose}>Close</button>
    </dialog>
  );
}

export function AppShell({ route, children }) {
  const { persistent } = useProgress();
  const [help, setHelp] = useState(false);
  const [noticeShut, setNoticeShut] = useState(false);

  useShortcuts({
    "/": () => goToSearch(),
    "?": () => setHelp((v) => !v),
  });

  const skip = (e) => {
    e.preventDefault();
    const main = document.querySelector("main");
    if (main) { main.setAttribute("tabindex", "-1"); main.focus(); }
  };

  return (
    <>
      <a href="#/" className="skip" onClick={skip}>Skip to content</a>
      <header className="appbar">
        <div className="appbar-inner">
          <a className="brand" href="#/">ML coding questions</a>
          <nav aria-label="Site" className="appnav">
            <button type="button" className="searchentry" onClick={goToSearch}>
              <span>Search</span><kbd aria-hidden="true">/</kbd>
            </button>
            <a href="#/progress" aria-current={route.page === "progress" ? "page" : undefined}>Progress</a>
            <button type="button" className="linklike" onClick={() => setHelp(true)} aria-label="Keyboard shortcuts">?</button>
          </nav>
          <ThemeSwitch />
        </div>
      </header>
      {!persistent && !noticeShut && (
        <p role="status" className="notice">
          Progress is not being saved in this browser. Storage is blocked or full, so done and bookmark marks last only until you close the page.
          <button type="button" className="linklike" onClick={() => setNoticeShut(true)}>Dismiss</button>
        </p>
      )}
      {children}
      <ShortcutHelp open={help} onClose={() => setHelp(false)} />
    </>
  );
}
