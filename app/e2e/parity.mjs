// Compare search on a site with the local service's results saved in parity_reference.json.
// Run: SITE=https://harsha-moparthy.github.io/interml/ node parity.mjs
// Per query: overlap of the top 10 with the reference, scaled to 10 when the reference has fewer results.
// Fails when the mean is below MIN_MEAN (default 7) or a query with reference results shows none.
// BLOCK="<glob>" aborts matching requests, to prove the check fails when search is broken.
import { readFileSync } from "node:fs";
import { chromium } from "playwright-core";
const SITE = process.env.SITE;
if (!SITE) { console.log("FAIL set SITE to the address of the site"); process.exit(1); }
const MIN_MEAN = Number(process.env.MIN_MEAN || 7);
const ref = JSON.parse(readFileSync(new URL("./parity_reference.json", import.meta.url)));
const exe = process.env.CHROME || "/mnt/nas/harshamttl/.cache/ms-playwright/chromium-1148/chrome-linux/chrome";
const b = await chromium.launch({ executablePath: exe, args: ["--no-sandbox"] });
const p = await (await b.newContext({ viewport: { width: 1100, height: 800 } })).newPage();
if (process.env.BLOCK) await p.route(process.env.BLOCK, (r) => r.abort());
const t0 = Date.now();
try {
  await p.goto(SITE); await p.waitForSelector("article.result", { timeout: 30000 });
} catch (e) {
  console.log("FAIL the site did not show the question list: " + String(e).split("\n")[0]);
  await b.close(); process.exit(1);
}
const box = p.locator("input[type=search]");
const scores = []; const empties = []; let firstMs = null;
for (const [q, want] of Object.entries(ref.results)) {
  await box.fill(q);
  const t1 = Date.now();
  await p.waitForFunction(() => /· keywords \+ meaning|· keywords only/.test(document.body.innerText) || /No questions match/.test(document.body.innerText), null, { timeout: 120000 }).catch(() => {});
  await p.waitForTimeout(400);
  if (firstMs === null) firstMs = Date.now() - t1;
  const mode = /keywords \+ meaning/.test(await p.locator(".result-count").innerText()) ? "meaning" : "keywords";
  const got = (await p.locator(".rid").allInnerTexts()).slice(0, 10).map(Number);
  const denom = Math.min(10, want.top.length);
  const overlap = got.filter((i) => want.top.includes(i)).length;
  const score = (overlap / denom) * 10;
  scores.push(score);
  if (want.top.length && !got.length) empties.push(q);
  console.log(`${score.toFixed(1).padStart(4)}/10  [${mode}]  ${JSON.stringify(q)}\n        site ${got.join(",")}\n        ref  ${want.top.join(",")}`);
  await box.fill("");
  await p.waitForTimeout(150);
}
const mean = scores.reduce((a, c) => a + c, 0) / scores.length;
console.log(`\nqueries: ${scores.length}   mean overlap@10: ${mean.toFixed(2)}   worst: ${Math.min(...scores).toFixed(1)}   first search took ${(firstMs / 1000).toFixed(1)} s   total ${((Date.now() - t0) / 1000).toFixed(0)} s`);
console.log(`reference taken ${ref.taken} from ${ref.api}`);
const pass = mean >= MIN_MEAN && !empties.length;
console.log((pass ? "PASS" : "FAIL") + ` mean overlap ${mean.toFixed(2)} (needs ${MIN_MEAN})` + (empties.length ? `; empty lists for: ${empties.join(" | ")}` : ""));
if (!pass) process.exitCode = 1;
await b.close();
