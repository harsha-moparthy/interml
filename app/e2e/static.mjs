// Browser check for the published (static) copy: single origin, loading and fallback states of search by meaning.
// Run: SITE=http://127.0.0.1:8790/interml/ node static.mjs
import { chromium } from "playwright-core";
const SITE = process.env.SITE;
if (!SITE) { console.log("FAIL set SITE to the address of the static build"); process.exit(1); }
const exe = process.env.CHROME || "/mnt/nas/harshamttl/.cache/ms-playwright/chromium-1148/chrome-linux/chrome";
const b = await chromium.launch({ executablePath: exe, args: ["--no-sandbox"] });
const ok = (c, msg) => { console.log((c ? "PASS " : "FAIL ") + msg); if (!c) process.exitCode = 1; };
const page = async () => (await (await b.newContext({ viewport: { width: 1100, height: 800 } })).newPage());
const labelReady = (p) => p.waitForFunction(() => /· keywords \+ meaning|· keywords only/.test(document.body.innerText), null, { timeout: 90000 });
const errs = [];

// 1. A query typed while the model is still loading waits, then shows keywords + meaning, with one origin only.
{
  const p = await page();
  p.on("pageerror", (e) => errs.push(String(e)));
  const hosts = new Set(); p.on("request", (r) => { if (/^https?:/.test(r.url())) hosts.add(new URL(r.url()).host); });
  await p.route("**/models/**", async (r) => { await new Promise((ok) => setTimeout(ok, 3000)); r.continue(); });   // keep the loading state long enough to see
  await p.goto(SITE); await p.waitForSelector("article.result");
  ok(await p.locator("[data-testid=meaning-loading]").isVisible(), "loading note shows while the model loads");
  await p.fill("input[type=search]", "make training use less memory");
  await p.waitForTimeout(500);
  ok(/Searching for/.test(await p.locator(".result-count").innerText()), "status line says Searching while waiting, not a stale count");
  await labelReady(p);
  ok((await p.locator(".result-count").innerText()).includes("keywords + meaning"), "list ends as keywords + meaning without a second action");
  ok((await p.locator("[data-testid=meaning-loading]").count()) === 0, "loading note is gone once ready");
  const ids = (await p.locator(".rid").allInnerTexts()).map(Number);
  ok(ids.some((i) => [852, 710, 561, 563, 562, 57].includes(i)), `memory-saving questions are found without shared words (${ids.slice(0, 5)})`);
  ok([...hosts].every((h) => h === new URL(SITE).host), `every request went to the site's own host (${[...hosts]})`);
}

// 2. Meaning model unavailable: keywords only, and the page says so.
{
  const p = await page();
  p.on("pageerror", (e) => errs.push(String(e)));
  await p.route("**/models/**", (r) => r.abort());
  await p.goto(SITE); await p.waitForSelector("article.result");
  await p.fill("input[type=search]", "softmax attention");
  await labelReady(p);
  ok((await p.locator(".result-count").innerText()).includes("keywords only"), "blocked model: list says keywords only");
  ok((await p.locator("p.sub", { hasText: "could not load" }).count()) === 1, "blocked model: message explains it");
  ok((await p.locator("article.result").count()) > 0, "blocked model: keyword results still shown");
}

// 3. Operators work without the model.
{
  const p = await page();
  await p.route("**/models/**", (r) => r.abort());
  await p.goto(SITE); await p.waitForSelector("article.result");
  await p.fill("input[type=search]", "chapter:a03 kv cache");
  await labelReady(p);
  const ids = (await p.locator(".rid").allInnerTexts()).map(Number);
  ok(ids.length > 0 && ids.every((i) => i >= 685 && i <= 736), `chapter:a03 keeps results inside A03 (${ids.length} shown)`);
}
console.log(errs.length ? "console/page errors: " + errs.join(" | ") : "console/page errors: none");
await b.close();
