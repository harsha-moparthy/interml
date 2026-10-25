import { useEffect, useState } from "react";
import { ArticlePage } from "./Article.jsx";
import { IndexPage } from "./Index.jsx";
import { ProgressPage } from "./Progress.jsx";
import { AppShell } from "./Shell.jsx";
import { QuestionPage } from "./Question.jsx";

function parseRoute(hash) {
  let m = hash.match(/^#\/question\/(\d+)$/);
  if (m) return { page: "question", id: Number(m[1]) };
  m = hash.match(/^#\/article\/([a-z0-9-]+)$/);
  if (m) return { page: "article", id: m[1] };
  if (/^#\/progress$/.test(hash)) return { page: "progress" };
  m = hash.match(/^#\/lesson-db\/([a-z0-9-]+)$/);   // the database-backed lesson, even when a hand-built page owns the usual URL
  if (m) return { page: "article", id: m[1], raw: true };
  return { page: "index" };
}

function useRoute() {
  const [route, setRoute] = useState(() => parseRoute(window.location.hash));
  useEffect(() => {
    const on = () => setRoute(parseRoute(window.location.hash));
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export default function App() {
  const route = useRoute();
  const [filters, setFilters] = useState({ query: "", chapter: "", lesson: "", state: "all" });
  let page;
  if (route.page === "question") page = <QuestionPage id={route.id} />;
  else if (route.page === "article") page = <ArticlePage id={route.id} />;
  else if (route.page === "progress") page = <ProgressPage />;
  else page = <IndexPage filters={filters} setFilters={setFilters} />;
  return <AppShell route={route}>{page}</AppShell>;
}
