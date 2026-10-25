// Browser checks for the interface layer: shell, theme, shortcuts, progress, navigation (change redesign-learning-ui).
// Run against real servers (`node ui.mjs`) or the fixture servers (`./run_suites.sh ui`). Needs playwright-core.
import { chromium } from "playwright-core";
const PORT = process.env.WEB_PORT || 5173;
const BASE = process.env.SITE || `http://127.0.0.1:${PORT}/`;   // SITE=<published address> runs this suite against the static copy
const api = async (path) => (await fetch(process.env.SITE ? `${BASE}static-api/${path}.json` : `${BASE}api/${path}`)).json();
const exe = "/mnt/nas/harshamttl/.cache/ms-playwright/chromium-1148/chrome-linux/chrome";
const b = await chromium.launch({ executablePath: exe, args: ["--no-sandbox"] });
const errs = [];
const ok = (c, msg) => { console.log((c ? "PASS " : "FAIL ") + msg); if (!c) process.exitCode = 1; };

async function open({ scheme = "light", width = 1100, height = 800, storage = true } = {}) {
  const ctx = await b.newContext({ viewport: { width, height }, colorScheme: scheme });
  if (!storage) {
    await ctx.addInitScript(() => {
      const deny = () => { throw new DOMException("blocked", "SecurityError"); };
      Object.defineProperty(window, "localStorage", { get: deny });
    });
  }
  const p = await ctx.newPage();
  p.on("pageerror", (e) => errs.push(String(e)));
  // On the published copy the meaning model preloads; a page closed mid-download makes Chromium log this about the aborted wasm fetch.
  const abortedWasm = /^(wasm streaming compile failed|falling back to ArrayBuffer instantiation)/;
  p.on("console", (m) => { if (m.type() === "error" && !/Failed to load resource/.test(m.text()) && !(process.env.SITE && abortedWasm.test(m.text()))) errs.push(m.text()); });
  return { ctx, p };
}
const bg = (p) => p.evaluate(() => getComputedStyle(document.body).backgroundColor);
const attr = (p) => p.evaluate(() => document.documentElement.getAttribute("data-theme"));
const articles = await api("articles");
const lesson = articles.find((a) => a.kind === "lesson")?.id ?? articles[0].id;

// ---------------------------------------------------------------- shell: theme
{
  const { ctx, p } = await open({ scheme: "dark" });
  await p.goto(BASE); await p.waitForSelector(".appbar");
  ok((await attr(p)) === null, "system theme sets no data-theme");
  ok((await bg(p)) === "rgb(16, 19, 23)", "system dark shows the dark background");
  await p.getByRole("button", { name: "Light", exact: true }).click();
  ok((await attr(p)) === "light" && (await bg(p)) === "rgb(250, 249, 247)", "choosing Light switches the page to light");
  ok((await p.evaluate(() => localStorage.getItem("placement.theme"))) === "light", "choice is saved");
  await p.reload(); await p.waitForSelector(".appbar");
  ok((await attr(p)) === "light" && (await bg(p)) === "rgb(250, 249, 247)", "light choice survives a reload while the system prefers dark");
  ok((await p.getByRole("button", { name: "Light", exact: true }).getAttribute("aria-pressed")) === "true", "switch shows the saved choice");
  await p.getByRole("button", { name: "System", exact: true }).click();
  ok((await attr(p)) === null && (await p.evaluate(() => localStorage.getItem("placement.theme"))) === null, "System removes the override and the saved value");
  await ctx.close();
}
{ // the saved choice is applied before the app code runs, so there is no flash
  const { ctx, p } = await open({ scheme: "dark" });
  await p.addInitScript(() => localStorage.setItem("placement.theme", "light"));
  await p.route("**/src/main.jsx", async (route) => { await new Promise((r) => setTimeout(r, 800)); route.continue(); });
  await p.goto(BASE, { waitUntil: "commit" });
  await p.waitForFunction(() => document.documentElement.hasAttribute("data-theme"), null, { timeout: 5000 });
  ok(!(await p.evaluate(() => !!document.querySelector(".appbar"))), "theme attribute is set while the app has not started yet");
  await ctx.close();
}
{ // blocked storage: the switch still works and nothing throws
  const { ctx, p } = await open({ scheme: "light", storage: false });
  await p.goto(BASE); await p.waitForSelector(".appbar");
  await p.getByRole("button", { name: "Dark", exact: true }).click();
  ok((await attr(p)) === "dark", "theme switch works when storage is blocked");
  await ctx.close();
}

// ---------------------------------------------------------------- shell: header, routes, shortcuts
{
  const { ctx, p } = await open();
  const routes = [["index", "#/", "h1"], ["question", "#/question/316", "h1 .id"], ["article", `#/article/${lesson}`, "h1"], ["progress", "#/progress", "h1"]];
  for (const [name, hash, ready] of routes) {
    await p.goto(BASE + hash); await p.waitForSelector(ready);
    ok(await p.locator("header.appbar a.brand").isVisible(), `header is shown on the ${name} page`);
    ok((await p.locator("header.appbar a.brand").getAttribute("href")) === "#/", `site name links to the index on the ${name} page`);
  }
  ok((await p.locator("header.appbar a[href='#/progress']").count()) === 1, "header links to progress");

  await p.goto(BASE + "#/question/300"); await p.waitForSelector("h1 .id");
  ok((await p.locator("h1").innerText()).includes("300"), "old link #/question/300 opens question 300");
  await p.goto(BASE + `#/lesson-db/${lesson}`); await p.waitForSelector("h1");
  ok(await p.locator("main.article").isVisible(), "old link #/lesson-db/ID opens an article");
  await p.goto(BASE + "#/progress"); await p.waitForSelector("h1");
  ok((await p.locator("h1").innerText()) === "Progress", "#/progress opens the progress page");

  await p.goto(BASE + "#/question/316"); await p.waitForSelector("h1 .id");
  await p.getByRole("button", { name: /^Search/ }).click();
  await p.waitForSelector("article.result");
  ok(await p.evaluate(() => document.activeElement?.type === "search"), "header Search from a question page opens the index with search focused");

  await p.goto(BASE + "#/question/316"); await p.waitForSelector("h1 .id");
  await p.locator("h1").click();
  await p.keyboard.press("/");
  await p.waitForSelector("article.result");
  ok(await p.evaluate(() => document.activeElement?.type === "search"), "/ goes to search from another page");

  const box = p.locator("input[type=search]");
  await box.fill("");
  await p.keyboard.type("a/b?");
  ok((await box.inputValue()) === "a/b?", "typing / and ? in the search box enters the characters");
  ok(!(await p.locator("dialog[open]").count()), "typing ? in a field does not open the help");
  await p.keyboard.press("Escape");
  ok((await box.inputValue()) === "", "Escape clears the search");

  await p.locator("h1").click();   // leave the field
  await p.keyboard.press("?");
  await p.waitForSelector("dialog[open]");
  const rows = await p.locator("dialog[open] dl > div").count();
  ok(rows >= 3, `help overlay lists the shortcuts (${rows})`);
  await p.keyboard.press("Escape");
  ok((await p.locator("dialog[open]").count()) === 0, "Escape closes the help overlay");
  await ctx.close();
}

// ---------------------------------------------------------------- shell: waiting and failed states
{
  const { ctx, p } = await open();
  let hits = 0, mode = "slow";
  await p.route("**/{api/exercises/316,static-api/exercises/316.json}", async (route) => {
    hits++;
    if (mode === "slow") { await new Promise((r) => setTimeout(r, 700)); return route.continue(); }
    if (mode === "fail") return route.fulfill({ status: 503, contentType: "text/plain", body: "down" });
    return route.continue();
  });
  await p.goto(BASE + "#/question/316");
  await p.waitForSelector(".skeleton");
  ok((await p.locator(".skeleton").getAttribute("aria-busy")) === "true", "slow load shows a placeholder, not a blank page");
  await p.waitForSelector("h1 .id");
  ok((await p.locator(".skeleton").count()) === 0, "placeholder goes away when the data arrives");

  mode = "fail"; hits = 0;
  await p.goto(BASE + "#/question/316"); await p.goto(BASE + "#/");
  await p.goto(BASE + "#/question/316");
  await p.reload();
  await p.waitForSelector(".error-notice");
  const msg = await p.locator(".error-notice").innerText();
  ok(/Cannot load question 316/.test(msg) && /503/.test(msg), "failure names what failed and why");
  const before = hits;
  mode = "ok";
  await p.getByRole("button", { name: "Try again" }).click();
  await p.waitForSelector("h1 .id");
  ok(hits === before + 1, "Try again repeats the request once");
  ok((await p.locator(".error-notice").count()) === 0, "page recovers after a successful retry");

  // reduced motion: no shimmer
  await p.emulateMedia({ reducedMotion: "reduce" });
  mode = "slow";
  await p.goto(BASE + "#/question/317");
  await p.route("**/{api/exercises/317,static-api/exercises/317.json}", async (route) => { await new Promise((r) => setTimeout(r, 600)); route.continue(); });
  await p.reload();
  await p.waitForSelector(".skeleton-line");
  const anim = await p.locator(".skeleton-line").first().evaluate((e) => getComputedStyle(e).animationName + "|" + getComputedStyle(e).animationDuration);
  ok(/none|0s/.test(anim), `skeleton does not animate under reduced motion (${anim})`);
  await ctx.close();
}

// ---------------------------------------------------------------- shell: motion and phone width
{
  const { ctx, p } = await open({ width: 360, height: 780 });
  await p.emulateMedia({ reducedMotion: "reduce" });
  const pages = [["index", "#/", "article.result"], ["question", "#/question/316", "h1 .id"], ["article", `#/article/${lesson}`, "main.article h1"], ["progress", "#/progress", "h1"]];
  for (const [name, hash, ready] of pages) {
    await p.goto(BASE + hash); await p.waitForSelector(ready);
    await p.waitForTimeout(300);
    const m = await p.evaluate(() => {
      const de = document.documentElement;
      const hdr = [...document.querySelectorAll(".appbar a, .appbar button")].map((e) => e.getBoundingClientRect());
      return { over: de.scrollWidth - de.clientWidth, running: document.getAnimations().filter((a) => a.playState === "running").length,
        tr: getComputedStyle(document.querySelector(".appbar button")).transitionDuration, hdrRight: Math.max(...hdr.map((r) => r.right)), hdrLeft: Math.min(...hdr.map((r) => r.left)), hdrH: document.querySelector(".appbar").getBoundingClientRect().height };
    });
    ok(m.over <= 0, `${name} page has no horizontal scroll at 360 px (overflow ${m.over}px)`);
    ok(m.hdrRight <= 360 && m.hdrLeft >= 0, `${name} page header controls stay inside 360 px`);
    ok(m.hdrH <= 100, `${name} page header uses at most two rows at 360 px (${Math.round(m.hdrH)}px tall)`);
    ok(m.running === 0, `${name} page runs no animation under reduced motion (${m.running})`);
    ok(/^0s/.test(m.tr), `${name} page has no transitions under reduced motion (${m.tr})`);
  }
  await ctx.close();
}

// ---------------------------------------------------------------- index
{
  const { ctx, p } = await open();
  const allReqs = [];
  p.on("request", (r) => { if (/(\/api\/search\?q=&chapter=&lesson=&limit=1100|\/static-api\/all\.json)$/.test(r.url())) allReqs.push(r.url()); });
  await p.goto(BASE); await p.waitForSelector("article.result");
  const summary = await p.locator(".summary").innerText();
  ok(/0\s+of\s+1100 questions done/.test(summary), "first visit: summary shows 0 of 1100 done");
  ok((await p.locator(".cta").getAttribute("href")) === "#/question/1", "first visit: start action opens question 1");
  ok((await p.locator("article.result").count()) === 1100, "index lists 1,100 questions");
  await p.locator("a.result-link").first().click(); await p.waitForSelector("h1 .id");
  await p.goBack(); await p.waitForSelector("article.result");
  ok(allReqs.length === 1, `the full list is requested once across index, question and back (${allReqs.length})`);

  // chapter navigation
  const nav = p.locator("nav.chapters");
  ok(await nav.isVisible(), "chapter list is shown on a wide screen");
  await nav.getByRole("button", { name: /^C05/ }).click();
  await p.waitForFunction(() => document.querySelectorAll("article.result").length < 1100);
  const metas = await p.locator("article.result .rmeta span:first-child").allInnerTexts();
  ok(metas.length > 0 && metas.every((t) => t.startsWith("C05")), `choosing C05 lists only C05 questions (${metas.length})`);
  ok((await nav.getByRole("button", { name: /^C05/ }).getAttribute("aria-current")) === "true", "active chapter is marked");
  const c05 = nav.getByRole("button", { name: /^C05/ });
  ok(/0 of \d+/.test(await c05.innerText()), "chapter shows 0 done before any marks");
  await p.locator("article.result").first().getByRole("button", { name: /done/i }).click();
  ok(/^1 of|\n1 of /.test(await c05.innerText()) || /1 of \d+/.test(await c05.innerText()), "chapter count rises after marking a question done");
  ok((await c05.getByRole("progressbar").getAttribute("aria-valuenow")) === "1", "chapter progress bar follows");
  ok(/1\s+of\s+1100 questions done/.test(await p.locator(".summary").innerText()), "summary count follows");
  const card = p.locator("article.result").first();
  ok(/✓ Done/.test(await card.innerText()), "done card says Done in text, not only colour");
  await card.getByRole("button", { name: /bookmark/i }).click();
  ok(/★ Bookmarked/.test(await card.innerText()), "bookmarked card says Bookmarked in text");

  // quick filters
  await p.getByRole("button", { name: "Clear", exact: true }).click();
  await p.getByRole("button", { name: "Done", exact: true }).click();
  await p.waitForFunction(() => document.querySelectorAll("article.result").length === 1);
  ok(true, "Done filter lists only the done question");
  await p.getByRole("button", { name: "Not done", exact: true }).click();
  await p.waitForFunction(() => document.querySelectorAll("article.result").length === 1099);
  ok(true, "Not done filter lists the other 1,099");
  await p.getByRole("button", { name: "Bookmarked", exact: true }).click();
  await p.waitForFunction(() => document.querySelectorAll("article.result").length === 1);
  ok(true, "Bookmarked filter lists the bookmarked question");
  await p.locator("article.result").first().getByRole("button", { name: /bookmark/i }).click();
  await p.waitForSelector(".empty");
  ok(/No bookmarks yet/.test(await p.locator(".empty").innerText()), "empty bookmark filter explains how to bookmark");

  // search still works with a learner filter, and Clear resets everything
  await p.getByRole("button", { name: "All", exact: true }).click();
  await p.fill("input[type=search]", "attention");
  await p.waitForFunction(() => { const n = document.querySelectorAll("article.result").length; return n > 0 && n <= 40; });
  ok(true, "search shows a first page of results");
  await p.getByRole("button", { name: "Clear", exact: true }).click();
  ok((await p.inputValue("input[type=search]")) === "", "Clear empties the query");
  await ctx.close();
}
{ // returning learner: resume action
  const { ctx, p } = await open();
  await p.addInitScript(() => { if (!localStorage.getItem("placement.progress.v1")) localStorage.setItem("placement.progress.v1", JSON.stringify({ v: 1, done: {}, bookmarks: {}, reading: {}, recent: [{ kind: "question", id: 316, t: Date.now() }] })); });
  await p.goto(BASE); await p.waitForSelector(".summary .cta");
  ok((await p.locator(".cta").getAttribute("href")) === "#/question/316", "returning learner: resume opens the last question");
  await ctx.close();
}
{ // scroll position and filters survive a visit to a question
  const { ctx, p } = await open({ height: 800 });
  await p.goto(BASE); await p.waitForSelector("article.result");
  await p.fill("input[type=search]", "attention");
  await p.waitForFunction(() => { const n = document.querySelectorAll("article.result").length; return n > 5 && n <= 40; });
  await p.mouse.wheel(0, 700); await p.waitForTimeout(300);
  const link = p.locator("article.result a.result-link").nth(3);
  await link.scrollIntoViewIfNeeded();
  const y = await p.evaluate(() => window.scrollY);
  await link.click(); await p.waitForSelector("h1 .id");
  await p.goBack(); await p.waitForSelector("article.result");
  await p.waitForTimeout(300);
  ok((await p.inputValue("input[type=search]")) === "attention", "query is kept on return");
  const y2 = await p.evaluate(() => window.scrollY);
  ok(y > 100 && Math.abs(y2 - y) < 60, `scroll position is restored (${y} -> ${y2})`);
  await ctx.close();
}
for (const w of [360, 768, 1280]) { // index layout at three widths
  const { ctx, p } = await open({ width: w });
  await p.goto(BASE); await p.waitForSelector("article.result");
  const over = await p.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  ok(over <= 0, `index has no horizontal scroll at ${w} px (overflow ${over}px)`);
  const sel = await p.locator(".chapter-select").isVisible(), nav = await p.locator("nav.chapters").isVisible();
  ok(w >= 900 ? nav && !sel : !nav && sel, `chapter navigation fits ${w} px (${nav ? "list" : "select"})`);
  await ctx.close();
}

// ---------------------------------------------------------------- question page
const lessonHit = async (kind) => (process.env.SITE ? (await api("all")).items.find((e) => e.lesson === kind)?.id : (await api(`search?q=&chapter=&lesson=${kind}&limit=1`)).items?.[0]?.id);
const LQ = (await lessonHit("complete")) ?? (await lessonHit("planned")) ?? 316;   // a question that has a linked lesson
let eqId = 316;
for (const id of [4, 19, 25, 60, 124, 200, 301, 316, 340, 410]) {
  const d = await api(`exercises/${id}`);
  if (/\$\$/.test(d.prompt)) { eqId = id; break; }
}
{
  const { ctx, p } = await open({ height: 900 });
  const solReqs = [];
  p.on("request", (r) => { if (/\/solution(\.json)?$/.test(r.url())) solReqs.push(r.url()); });
  await p.goto(BASE + `#/question/${LQ}`); await p.waitForSelector("h1 .id");
  const top = await p.evaluate(() => ["h1", ".qactions", ".question .meta"].map((s) => document.querySelector(s).getBoundingClientRect().bottom));
  ok(top.every((y) => y < 900), "header, actions and meta are visible without scrolling");
  ok(/C\d+/.test(await p.locator(".question .meta").innerText()) && /hard|medium|easy/i.test(await p.locator(".question .meta").innerText()), "meta shows chapter and difficulty");
  const lessonY = await p.locator(".lesson-card").first().evaluate((e) => e.getBoundingClientRect().top);
  const taskY = await p.locator("#task-h").evaluate((e) => e.getBoundingClientRect().top);
  const readY = await p.locator("#reading-h").evaluate((e) => e.getBoundingClientRect().top);
  ok(lessonY < taskY && lessonY < readY, "lesson card comes before the task and the readings");
  ok(solReqs.length === 0, "no solution request before the learner asks");
  await p.getByRole("button", { name: "Show solution" }).click();
  await p.waitForSelector(".cm-content");
  ok(solReqs.length === 1, "solution is requested once, after the click");
  ok((await p.locator(".cm-content .tok-keyword").count()) > 0, "solution code is syntax highlighted");
  ok((await p.locator(".readings li").count()) === 2, "two readings are listed");
  ok(/[A-Za-z]/.test(await p.locator(".readings li .reading-meta").first().innerText()), "reading shows publisher and access note");

  await p.goto(BASE + "#/question/1"); await p.waitForSelector("h1 .id");
  const q1 = await api("exercises/1");
  ok((await p.locator(".lesson-card").count()) === q1.articles.length, `lesson cards match the linked articles (${q1.articles.length}) on question 1`);
  ok((await p.locator(".pager-off").first().innerText()).includes("first"), "first question shows its missing direction as unavailable");
  const hashBefore = await p.evaluate(() => location.hash);
  await p.locator("h1").click(); await p.keyboard.press("ArrowLeft");
  ok((await p.evaluate(() => location.hash)) === hashBefore, "left arrow does nothing on the first question");
  await p.keyboard.press("ArrowRight");
  await p.waitForFunction(() => location.hash === "#/question/2");
  ok(true, "right arrow opens the next question");
  await p.goto(BASE + "#/question/1100"); await p.waitForSelector("h1 .id");
  ok((await p.locator(".pager-off").last().innerText()).includes("last"), "last question shows its missing direction as unavailable");
  await p.locator("h1").click(); await p.keyboard.press("ArrowRight");
  ok((await p.evaluate(() => location.hash)) === "#/question/1100", "right arrow does nothing on the last question");

  await p.goto(BASE + `#/question/${LQ}`); await p.waitForSelector(".pager-pos");
  ok(/\d+ of \d+ in C\d+/.test(await p.locator(".pager-pos").innerText()), "page shows the position in the chapter");
  await p.evaluate(() => { const i = document.createElement("input"); i.id = "tmp"; document.body.appendChild(i); i.focus(); });
  await p.keyboard.press("ArrowRight");
  ok((await p.evaluate(() => location.hash)) === `#/question/${LQ}`, "arrow keys do nothing inside a field");
  await p.evaluate(() => document.getElementById("tmp").remove());

  // done and bookmark here match the card on the index
  await p.getByRole("button", { name: "Mark done" }).click();
  ok((await p.getByRole("button", { name: /✓ Done/ }).getAttribute("aria-pressed")) === "true", "Mark done toggles on the question page");
  await p.getByRole("button", { name: /Bookmark/ }).click();
  await p.goto(BASE + "#/"); await p.waitForSelector("article.result");
  const card = p.locator("article.result", { has: p.locator(`a[href='#/question/${LQ}']`) });
  const t = await card.innerText();
  ok(/✓ Done/.test(t) && /★ Bookmarked/.test(t), "the index card shows what was set on the question page");

  // recent: twelve distinct questions keep the latest ten, newest first
  for (let id = 501; id <= 512; id++) { await p.goto(BASE + `#/question/${id}`); await p.waitForSelector("h1 .id"); await p.waitForFunction((n) => document.querySelector("h1").innerText.includes(String(n)), id); }
  const recent = await p.evaluate(() => JSON.parse(localStorage.getItem("placement.progress.v1")).recent.map((r) => r.id));
  ok(recent.length === 10 && recent[0] === 512 && recent[9] === 503, `recent keeps the latest ten, newest first (${recent.join(",")})`);
  await ctx.close();
}
{ // solution failure and long equations on a phone
  const { ctx, p } = await open();
  let fail = true;
  await p.route("**/{api/exercises/317/solution,static-api/exercises/317/solution.json}", (route) => (fail ? route.fulfill({ status: 500, contentType: "text/plain", body: "x" }) : route.continue()));
  await p.goto(BASE + "#/question/317"); await p.waitForSelector("h1 .id");
  await p.getByRole("button", { name: "Show solution" }).click();
  await p.waitForSelector("#solution-panel .error-notice");
  ok(/solution/.test(await p.locator("#solution-panel .error-notice").innerText()) && (await p.locator("h2#task-h").isVisible()), "solution failure is explained and the rest of the page stays usable");
  fail = false;
  await p.getByRole("button", { name: "Try again" }).click();
  await p.waitForSelector(".cm-content");
  ok(true, "retry loads the solution");
  await ctx.close();
}
{
  const { ctx, p } = await open({ width: 360 });
  await p.goto(BASE + `#/question/${eqId}`); await p.waitForSelector("h1 .id");
  const m = await p.evaluate(() => ({ over: document.documentElement.scrollWidth - document.documentElement.clientWidth, eq: document.querySelectorAll(".prompt-equation").length }));
  ok(m.over <= 0, `question ${eqId} (${m.eq} display equations) has no page scroll at 360 px`);
  await ctx.close();
}

// ---------------------------------------------------------------- article reading
let rich = { id: lesson, h2: 0, blocks: 0 };
for (const a of articles) {
  const d = await api(`articles/${a.id}`);
  const h2 = d.blocks.filter((b) => b.type === "heading" && b.level === 2).length;
  if (h2 > rich.h2 || (h2 === rich.h2 && d.blocks.length > rich.blocks)) rich = { id: a.id, h2, blocks: d.blocks.length, prereqs: d.prerequisites.length, exercises: d.exercises.length, kind: d.kind };
}
const RICH = `#/article/${rich.id}`;
const ratioIn = (p, sel) => p.evaluate((sel) => {
  const parse = (c) => c.match(/[\d.]+/g).map(Number);
  const lum = ([r, g, b]) => { const f = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; }; return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b); };
  const el = document.querySelector(sel); if (!el) return null;
  let bgEl = el, bg = [255, 255, 255];
  for (; bgEl; bgEl = bgEl.parentElement) { const c = parseVal(getComputedStyle(bgEl).backgroundColor); if (c && c[3] !== 0) { bg = c; break; } }
  function parseVal(c) { const m = c.match(/[\d.]+/g); return m ? m.map(Number).concat(m.length === 3 ? [1] : []) : null; }
  const fg = parse(getComputedStyle(el).color);
  const [a, b] = [lum(fg), lum(bg)];
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}, sel);
ok(rich.h2 > 2, `found an article with several sections for the reading checks (${rich.id}: ${rich.h2} sections)`);
{
  const { ctx, p } = await open({ width: 1280, height: 800 });
  await p.emulateMedia({ reducedMotion: "reduce" });
  await p.goto(BASE + RICH); await p.waitForSelector(".toc-list a");
  ok(await p.locator(".toc-list").isVisible() && !(await p.locator(".toc-toggle").isVisible()), "wide screen: table of contents is always open beside the text");
  const n = await p.locator(".toc-list a").count();
  ok(n === rich.h2, `table of contents has one link per section (${n})`);
  const pos = await p.evaluate(() => { const t = document.querySelector(".toc").getBoundingClientRect(), b = document.querySelector(".article-body").getBoundingClientRect(); return [t.right <= b.left + 1, getComputedStyle(document.querySelector(".toc")).position]; });
  ok(pos[0] && pos[1] === "sticky", "table of contents is a sticky column left of the text");
  const target = Math.min(2, n - 1);
  const link = p.locator(".toc-list a").nth(target);
  const headId = await p.evaluate((i) => [...document.querySelectorAll(".article-body h2[id]")][i].id, target);
  await link.click();
  await p.waitForTimeout(250);
  const where = await p.evaluate((id) => ({ top: document.getElementById(id).getBoundingClientRect().top, focused: document.activeElement.id, hash: location.hash, bottom: scrollY + innerHeight >= document.documentElement.scrollHeight - 2 }), headId);
  ok(where.top >= 0 && (where.top < 250 || where.bottom), `choosing an entry scrolls to its section (heading at ${Math.round(where.top)}px${where.bottom ? ", page end reached" : ""})`);
  ok(where.focused === headId, "focus moves to the section heading");
  ok(where.hash === RICH, "the page route is unchanged by a contents link");
  ok((await link.getAttribute("aria-current")) === "location", "the entry for the section in view is marked current");
  await p.evaluate(() => scrollTo(0, 0)); await p.waitForTimeout(250);
  const before = await p.locator(".toc-list a[aria-current=location]").innerText();
  await p.evaluate(() => { const h = document.querySelectorAll(".article-body h2"); h[h.length - 1].scrollIntoView(); });
  await p.waitForTimeout(300);
  const after = await p.locator(".toc-list a[aria-current=location]").innerText();
  ok(before !== after && (await p.locator(".toc-list a[aria-current=location]").count()) === 1, `current section follows scrolling ("${before.trim().slice(0, 20)}" to "${after.trim().slice(0, 20)}")`);
  await ctx.close();
}
{ // phone: the contents open from a button and never cover the text when closed
  const { ctx, p } = await open({ width: 360, height: 780 });
  await p.emulateMedia({ reducedMotion: "reduce" });
  await p.goto(BASE + RICH); await p.waitForSelector(".toc-toggle");
  ok(!(await p.locator(".toc-list").isVisible()), "phone: contents start closed");
  const covered = await p.evaluate(() => { const t = document.querySelector(".toc").getBoundingClientRect(), b = document.querySelector(".article-body").getBoundingClientRect(); return t.bottom > b.top + 1 && getComputedStyle(document.querySelector(".toc")).position !== "static"; });
  ok(!covered, "phone: the closed contents do not overlap the text");
  await p.locator(".toc-toggle").click();
  ok((await p.locator(".toc-toggle").getAttribute("aria-expanded")) === "true" && (await p.locator(".toc-list").isVisible()), "phone: the button opens the contents");
  await p.locator(".toc-list a").nth(1).click(); await p.waitForTimeout(200);
  ok(!(await p.locator(".toc-list").isVisible()), "phone: choosing an entry closes the contents");
  ok((await p.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)) <= 0, "phone: article has no horizontal scroll");
  await ctx.close();
}
{ // reading progress and continue
  const { ctx, p } = await open({ width: 1100, height: 800 });
  await p.emulateMedia({ reducedMotion: "reduce" });
  await p.goto(BASE + RICH); await p.waitForSelector(".readbar");
  ok((await p.locator(".readbar").getAttribute("aria-valuenow")) === "0", "reading progress starts at 0");
  const room = await p.evaluate(() => document.documentElement.scrollHeight - innerHeight);
  await p.evaluate((y) => scrollTo(0, y), room / 2); await p.waitForTimeout(700);
  const now = Number(await p.locator(".readbar").getAttribute("aria-valuenow"));
  ok(now >= 45 && now <= 55, `halfway down shows about half (${now}%)`);
  const savedY = await p.evaluate((id) => JSON.parse(localStorage.getItem("placement.progress.v1")).reading[id]?.y, rich.id);
  ok(savedY > 0.4 && savedY < 0.6, `position is saved (${savedY?.toFixed(2)})`);
  await p.reload(); await p.waitForSelector(".resume");
  ok(/about 5\d%/.test(await p.locator(".resume").innerText()), "a later visit offers to continue");
  await p.getByRole("button", { name: "Continue" }).click(); await p.waitForTimeout(400);
  const y2 = await p.evaluate(() => scrollY / (document.documentElement.scrollHeight - innerHeight));
  ok(Math.abs(y2 - savedY) < 0.05, `Continue returns to the saved place (${y2.toFixed(2)})`);
  ok((await p.locator(".resume").count()) === 0, "the continue offer goes away");
  await ctx.close();
}
{ // header, prerequisites, callouts and dark-theme legibility
  const { ctx, p } = await open({ width: 1100, height: 900, scheme: "dark" });
  await p.goto(BASE + RICH); await p.waitForSelector(".arthead");
  const head = await p.locator(".arthead .meta").innerText();
  ok(/Deep lesson|Prerequisite article/.test(head) && /Complete|Reviewed|Draft/.test(head) && /\d+ min read/.test(head), `article header shows kind, review state and reading time (${head.replace(/\n/g, " ")})`);
  ok((await p.locator(".pager a[href^='#/question/']").count()) === rich.exercises, "header links the supported questions");
  ok((await p.locator(".prereq a").count()) === rich.prereqs, `prerequisites are listed before the body (${rich.prereqs})`);
  const order = await p.evaluate(() => { const pr = document.querySelector(".prereq"), b = document.querySelector(".article-body"); return !pr || pr.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING; });
  ok(!!order, "prerequisites come before the first section");
  for (const [name, sel] of [["heading", ".article-body h2"], ["paragraph", ".article-body > p"], ["callout text", "aside.callout p"], ["callout label", "aside.callout > strong"], ["contents link", ".toc-list a"], ["equation", ".equation .eq"], ["check", ".check"], ["playground", ".playground h3"], ["code bar", ".codebar"]]) {
    const r = await ratioIn(p, sel);
    if (r === null) continue;
    ok(r >= 4.5, `dark theme: ${name} is legible (${r.toFixed(1)}:1)`);
  }
  const callout = await p.evaluate(() => { const c = document.querySelector("aside.callout"); return c ? { ico: !!c.querySelector(".ico"), label: c.querySelector("strong").innerText.trim() } : null; });
  ok(!callout || (callout.ico && callout.label.length > 1), "callouts carry an icon and a text label");
  const mistake = await p.evaluate(() => { const c = document.querySelector("aside.callout.mistake"); return c ? getComputedStyle(c).borderLeftStyle : null; });
  ok(mistake === null || mistake !== "solid", `mistake callouts differ in line style, not only colour (${mistake})`);
  await ctx.close();
}

{ // printing an article
  const { ctx, p } = await open({ width: 1100, height: 900, scheme: "dark" });
  await p.goto(BASE + RICH); await p.waitForSelector(".arthead");
  await p.emulateMedia({ media: "print", colorScheme: "dark" });
  const m = await p.evaluate(() => {
    const vis = (sel) => [...document.querySelectorAll(sel)].filter((e) => getComputedStyle(e).display !== "none").length;
    return { header: vis(".appbar"), toc: vis(".toc"), bar: vis(".readbar"), buttons: vis(".article button"), bg: getComputedStyle(document.body).backgroundColor, fg: getComputedStyle(document.querySelector(".article-body p")).color,
      figs: vis(".article-body figure, .article-body .equation"), text: document.querySelector(".article-body").innerText.length };
  });
  ok(m.header === 0 && m.toc === 0 && m.bar === 0, "print hides the header, table of contents and progress bar");
  ok(m.buttons === 0, `print hides interactive controls (${m.buttons} left)`);
  ok(m.bg === "rgb(255, 255, 255)" && /^rgb\(\d{1,2}, \d{1,2}, \d{1,2}\)$/.test(m.fg), `print is light and ink-friendly even from the dark theme (${m.bg} / ${m.fg})`);
  ok(m.figs > 0 && m.text > 500, `print keeps the text, figures and equations (${m.figs} figures, ${m.text} characters)`);
  const pdf = await p.pdf({ format: "A4", printBackground: false });
  ok(pdf.length > 5000, `a PDF can be produced (${Math.round(pdf.length / 1024)} KB)`);
  await ctx.close();
}

// ---------------------------------------------------------------- progress page
const SEED = { v: 1, done: { 1: 1000, 2: 1000, 3: 1000, 316: 1000 }, bookmarks: { 5: 2000, 316: 3000 }, reading: {},
  recent: [{ kind: "question", id: 316, t: Date.now() }, { kind: "article", id: lesson, t: Date.now() - 60000 }, { kind: "question", id: 2, t: Date.now() - 120000 }] };
const seedScript = (obj) => `localStorage.setItem("placement.progress.v1", ${JSON.stringify(JSON.stringify(obj))})`;
{
  const { ctx, p } = await open();
  await p.addInitScript(`if (!localStorage.getItem("placement.progress.v1")) { ${seedScript(SEED)} }`);
  await p.goto(BASE + "#/progress"); await p.waitForSelector(".pg-summary");
  ok(/4\s+of\s+1100 questions done/.test(await p.locator(".pg-summary").innerText()) && /2\s+bookmarked/.test(await p.locator(".pg-summary").innerText()), "totals show 4 of 1100 done and 2 bookmarked");
  ok((await p.locator(".pg-summary [role=progressbar]").getAttribute("aria-valuenow")) === "4", "overall bar matches");
  const c01 = p.locator(".pg-chapters li", { hasText: "C01" });
  ok(/3 of \d+/.test(await c01.innerText()), "chapter C01 shows 3 done");
  ok((await c01.getByRole("progressbar").getAttribute("aria-valuenow")) === "3", "chapter bar matches");
  ok((await c01.locator("a").getAttribute("href")) === "#/question/4", "chapter entry links to its next unfinished question");
  const bm = p.locator("#pg-bookmarks ~ ul a");
  ok((await bm.count()) === 2 && (await bm.first().getAttribute("href")) === "#/question/316", "bookmarks list newest first and link to their questions");
  const rc = p.locator("#pg-recent ~ ul a");
  const hrefs = await rc.evaluateAll((els) => els.map((e) => e.getAttribute("href")));
  ok(hrefs[0] === "#/question/316" && hrefs[1] === `#/article/${lesson}` && hrefs[2] === "#/question/2", `recent items link to questions and articles (${hrefs.join(" ")})`);
  ok(/Multi-Head|Attention|[A-Za-z]{4}/.test(await rc.first().innerText()), "recent items show titles, not only numbers");
  await p.locator("#pg-bookmarks ~ ul li").first().getByRole("button", { name: /Remove bookmark/ }).click();
  ok((await p.locator("#pg-bookmarks ~ ul li").count()) === 1, "a bookmark can be removed from this page");
  await rc.first().click(); await p.waitForSelector("h1 .id");
  ok(true, "a recent entry opens its question");
  await ctx.close();
}
{ // empty state, export / reset / import
  const { ctx, p } = await open();
  await p.goto(BASE + "#/progress"); await p.waitForSelector(".empty");
  ok(/Nothing here yet/.test(await p.locator(".empty").innerText()) && (await p.locator(".empty a.cta").getAttribute("href")) === "#/question/1", "empty progress explains how to start and links to question 1");
  ok(await p.getByRole("button", { name: /Reset all progress/ }).isDisabled(), "reset is disabled when there is nothing to reset");
  await p.evaluate((seed) => localStorage.setItem("placement.progress.v1", seed), JSON.stringify(SEED));
  await p.reload(); await p.waitForSelector(".pg-summary");
  const [dl] = await Promise.all([p.waitForEvent("download"), p.getByRole("button", { name: "Export progress" }).click()]);
  const text = await (await import("node:fs/promises")).readFile(await dl.path(), "utf8");
  const exported = JSON.parse(text);
  ok(/^placement-progress-\d{4}-\d\d-\d\d\.json$/.test(dl.suggestedFilename()) && exported.v === 1 && Object.keys(exported.done).length === 4, `export downloads a file with the progress (${dl.suggestedFilename()})`);

  await p.getByRole("button", { name: /Reset all progress/ }).click();
  ok(await p.getByRole("alertdialog").isVisible(), "reset asks for confirmation");
  ok(Object.keys(JSON.parse(await p.evaluate(() => localStorage.getItem("placement.progress.v1"))).done).length === 4, "nothing is deleted before the learner confirms");
  await p.getByRole("button", { name: "Cancel" }).click();
  ok(Object.keys(JSON.parse(await p.evaluate(() => localStorage.getItem("placement.progress.v1"))).done).length === 4 && !(await p.getByRole("alertdialog").count()), "Cancel keeps everything");
  await p.getByRole("button", { name: /Reset all progress/ }).click();
  await p.getByRole("button", { name: "Delete everything" }).click();
  await p.waitForSelector(".empty");
  ok((await p.evaluate(() => localStorage.getItem("placement.progress.v1"))) === null, "confirming deletes the stored progress");

  await p.setInputFiles("input[type=file]", { name: "bad.json", mimeType: "application/json", buffer: Buffer.from("{nope") });
  await p.waitForSelector("p.error");
  ok(/not valid JSON/.test(await p.locator("p.error").innerText()), "a bad file is rejected with a reason");
  await p.setInputFiles("input[type=file]", { name: "progress.json", mimeType: "application/json", buffer: Buffer.from(text) });
  await p.waitForSelector(".pg-summary");
  ok(/4\s+of\s+1100/.test(await p.locator(".pg-summary").innerText()) && /Imported 4 done/.test(await p.locator("[role=status].pg-ok").innerText()), "importing the export restores the progress");
  ok((await p.locator("#pg-bookmarks ~ ul li").count()) === 2, "bookmarks come back too");
  await ctx.close();
}
{ // progress data never leaves the browser
  const { ctx, p } = await open();
  const seen = [];
  p.on("request", (r) => seen.push({ method: r.method(), url: r.url(), body: r.postData() }));
  await p.goto(BASE); await p.waitForSelector("article.result");
  await p.locator("article.result").first().getByRole("button", { name: /done/i }).click();
  await p.locator("article.result").nth(1).getByRole("button", { name: /bookmark/i }).click();
  await p.goto(BASE + `#/article/${lesson}`); await p.waitForSelector(".arthead");
  await p.evaluate(() => scrollTo(0, 600)); await p.waitForTimeout(800);
  await p.goto(BASE + "#/progress"); await p.waitForSelector(".pg-summary");
  await p.getByRole("button", { name: "Export progress" }).click().catch(() => {});
  const bad = seen.filter((r) => r.method !== "GET" || r.body || !r.url.startsWith(BASE));
  ok(bad.length === 0, `marking, bookmarking and reading send no data anywhere (${seen.length} requests, ${bad.length} not plain local GETs)`);
  ok(!seen.filter((r) => /\/(api|media)\//.test(r.url)).some((r) => /progress|bookmark|done/i.test(new URL(r.url).pathname + new URL(r.url).search)), "no API request URL carries progress");
  await ctx.close();
}
{ // blocked storage and a corrupt record
  const { ctx, p } = await open({ storage: false });
  await p.goto(BASE); await p.waitForSelector("article.result");
  ok(/not being saved/.test(await p.locator(".notice").innerText()), "blocked storage: a notice says progress is not being saved");
  await p.locator("article.result").first().getByRole("button", { name: /done/i }).click();
  ok(/1\s+of\s+1100/.test(await p.locator(".summary").innerText()), "blocked storage: marking done still works for this visit");
  await p.goto(BASE + "#/progress"); await p.waitForSelector(".pg-summary");
  ok(true, "blocked storage: the progress page opens");
  await p.goto(BASE + "#/question/5"); await p.waitForSelector("h1 .id");
  ok(true, "blocked storage: question pages open");
  await ctx.close();
}
{
  const { ctx, p } = await open();
  await p.addInitScript(`localStorage.setItem("placement.progress.v1", "{this is not json")`);
  await p.goto(BASE); await p.waitForSelector("article.result");
  ok(/0\s+of\s+1100 questions done/.test(await p.locator(".summary").innerText()), "corrupt record: the site starts with empty progress");
  ok((await p.locator(".notice").count()) === 0, "corrupt record: no 'not saved' notice, because storage works");
  await p.locator("article.result").first().getByRole("button", { name: /done/i }).click();
  ok(JSON.parse(await p.evaluate(() => localStorage.getItem("placement.progress.v1"))).v === 1, "corrupt record: the next change replaces it with valid data");
  await ctx.close();
}

// ---------------------------------------------------------------- end
await b.close();
ok(errs.length === 0, "console errors: none" + (errs.length ? " -- " + errs.slice(0, 3).join(" | ") : ""));
