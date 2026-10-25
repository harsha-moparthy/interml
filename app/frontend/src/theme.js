// Theme choice: "light", "dark" or "system" (follow the operating system). Saved in this browser when storage allows it.
// index.html applies a saved light or dark choice before first paint, so the page does not flash.
import { useSyncExternalStore } from "react";

export const THEME_KEY = "placement.theme";
export const THEMES = ["light", "dark", "system"];

let choice = "system";
try {
  const saved = localStorage.getItem(THEME_KEY);
  if (saved === "light" || saved === "dark") choice = saved;
} catch { /* storage blocked: the choice still works for this visit */ }

const listeners = new Set();

function apply() {
  const root = document.documentElement;
  if (choice === "system") root.removeAttribute("data-theme"); else root.setAttribute("data-theme", choice);
}
apply();

export function setTheme(next) {
  if (!THEMES.includes(next)) return;
  choice = next;
  apply();
  try {
    if (next === "system") localStorage.removeItem(THEME_KEY); else localStorage.setItem(THEME_KEY, next);
  } catch { /* see above */ }
  listeners.forEach((fn) => fn());
}

const subscribe = (fn) => { listeners.add(fn); return () => listeners.delete(fn); };
export const useTheme = () => useSyncExternalStore(subscribe, () => choice);
