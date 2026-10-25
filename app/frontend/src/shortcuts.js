import { useEffect, useRef } from "react";

/** True when the key event comes from somewhere the learner is typing. */
export function isTypingTarget(el) {
  if (!el) return false;
  return /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName) || el.isContentEditable === true || !!el.closest?.(".cm-editor");
}

/**
 * Run `handlers[key]` for a plain key press outside fields. Modified keys (Ctrl, Cmd, Alt) are left to the browser.
 * A handler may return false to leave the event alone.
 */
export function useShortcuts(handlers, enabled = true) {
  const ref = useRef(handlers);
  ref.current = handlers;
  useEffect(() => {
    if (!enabled) return undefined;
    const on = (e) => {
      if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
      if (isTypingTarget(e.target) || isTypingTarget(document.activeElement)) return;
      const fn = ref.current[e.key];
      if (fn && fn(e) !== false) e.preventDefault();
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, [enabled]);
}

// "Search" can be asked for from any page. The index takes the request when it is on screen.
let searchRequested = false;
const searchListeners = new Set();
export function requestSearchFocus() { searchRequested = true; searchListeners.forEach((fn) => fn()); }
export function takeSearchRequest() { const was = searchRequested; searchRequested = false; return was; }
export function onSearchRequest(fn) { searchListeners.add(fn); return () => searchListeners.delete(fn); }
