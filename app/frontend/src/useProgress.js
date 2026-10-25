import { useSyncExternalStore } from "react";
import { progress } from "./progress.js";

/** The learner's progress; the component re-renders whenever it changes, in this tab or another. */
export function useProgress() {
  return useSyncExternalStore(progress.subscribe, progress.getSnapshot);
}
