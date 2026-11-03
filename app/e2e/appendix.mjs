// Browser check: the appendix chapters A01-A18 list their questions, by chapter list and by operator.
// Run: SITE=http://127.0.0.1:8790/interml/ node appendix.mjs     (default site: the dev server on WEB_PORT or 5173)
import fs from "node:fs";
import { chromium } from "playwright-core";
const SITE = process.env.SITE || `http://127.0.0.1:${process.env.WEB_PORT || 5173}/`;
const exe = process.env.CHROME || "/mnt/nas/harshamttl/.cache/ms-playwright/chromium-1148/chrome-linux/chrome";
const b = await chromium.launch({ executablePath: exe, args: ["--no-sandbox"] });
const p = await (await b.newContext({ viewport: { width: 1100, height: 800 } })).newPage();
const errs = []; p.on("pageerror", (e) => errs.push(String(e))); p.on("console", (m) => m.type() === "error" && errs.push(m.text()));
const ok = (c, msg) => { console.log((c ? "PASS " : "FAIL ") + msg); if (!c) process.exitCode = 1; };
const counts = { A01: 52, A02: 32, A03: 52, A04: 27, A05: 28, A06: 22, A07: 9, A08: 15, A09: 5, A10: 20, A11: 18, A12: 22, A13: 15, A14: 22, A15: 11, A16: 50, A17: 50, A18: 50 };
const additions=JSON.parse(fs.readFileSync(new URL("../content/interview_additions.json",import.meta.url)));
for (const item of additions) counts[`A${String(item.chapter_number).padStart(2,"0")}`]++;
const bankTotal=1100+additions.length;
// Searches show 40 results at a time plus a "Show 40 more (N left)" button, so the total is shown + left.
const total = () => document.querySelectorAll("article.result").length
  + Number(/\((\d+) left\)/.exec(document.body.innerText)?.[1] || 0);
const shown = () => p.evaluate(total);
const settle = (n) => p.waitForFunction(`(${total})() === ${n}`, null, { timeout: 15000 }).catch(() => {});
await p.goto(SITE);
await p.waitForSelector("article.result");
const pick = (id) => p.locator("nav.chapters button", { has: p.locator(`.cid:text-is("${id}")`) }).click();
for (const [id, n] of Object.entries(counts)) {
  await pick(id); await settle(n);
  ok((await shown()) === n, `chapter list: ${id} shows ${n} questions`);
}
await p.locator("nav.chapters button", { hasText: "All chapters" }).click(); await settle(bankTotal);
for (const [q, n] of [["chapter:a18", 50], ["chapter:A01", 52], ["ch:a03", counts.A03], ["chapter:5", 22]]) {
  await p.fill("input[type=search]", q); await settle(n);
  ok((await shown()) === n, `operator ${q} shows ${n} questions`);
}
await p.fill("input[type=search]", "chapter:a18");
await settle(50);
ok((await p.locator(".chips").innerText()).toLowerCase().includes("chapter: a18"), "chip shows chapter: a18");
// Narrow screens use the drop-down instead of the list.
await p.fill("input[type=search]", ""); await settle(bankTotal);
await p.setViewportSize({ width: 600, height: 900 });
const select = p.locator(".chapter-select select");
for (const id of ["A01", "A18"]) {
  await select.selectOption(id); await settle(counts[id]);
  ok((await shown()) === counts[id], `drop-down: ${id} shows ${counts[id]} questions`);
}
console.log(errs.length ? "console/page errors: " + errs.join(" | ") : "console/page errors: none");
await b.close();
