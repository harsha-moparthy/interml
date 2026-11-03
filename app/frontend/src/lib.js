import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "./staticApi.js";

export const pad = (n) => String(n).padStart(3, "0");

/** The complete question list. Every page that needs it shares one cached request. */
export const ALL_URL = `/api/search?${new URLSearchParams({ q: "", chapter: "", lesson: "", limit: 10000 })}`;
const cache = new Map();
const getJson = (url) => apiFetch(url).then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status} ${r.statusText}`))));

/**
 * Fetch JSON. `data` keeps the last result while a new url loads, so pages compare ids before showing it.
 * `loading` is true until the current url has answered. `retry()` clears an error and asks again.
 */
export function useFetch(url, { cache: cached = url === ALL_URL } = {}) {
  const [state, setState] = useState({ data: null, error: null, url: null });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let live = true;
    let pending;
    if (cached) {
      if (!cache.has(url)) cache.set(url, getJson(url).catch((e) => { cache.delete(url); throw e; }));
      pending = cache.get(url);
    } else pending = getJson(url);
    pending
      .then((data) => live && setState({ data, error: null, url }))
      .catch((error) => live && setState({ data: null, error, url }));
    return () => { live = false; };
  }, [url, attempt, cached]);
  const retry = useCallback(() => {
    setState((s) => ({ ...s, error: null, url: null }));
    setAttempt((n) => n + 1);
  }, []);
  return { data: state.data, error: state.error, loading: state.url !== url, retry };
}

export const useAllQuestions = () => useFetch(ALL_URL);
