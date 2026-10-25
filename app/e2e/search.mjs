// Browser check for search and the code viewer (real data, no fixtures needed).
import { chromium } from "playwright-core";
const exe = "/mnt/nas/harshamttl/.cache/ms-playwright/chromium-1148/chrome-linux/chrome";
const b = await chromium.launch({ executablePath: exe, args: ["--no-sandbox"] });
const ctx = await b.newContext({ viewport: { width: 1100, height: 900 }, permissions: ["clipboard-read", "clipboard-write"] });
const p = await ctx.newPage();
const errs = []; p.on("pageerror", (e) => errs.push(String(e))); p.on("console", (m) => m.type() === "error" && errs.push(m.text()));
const ok = (c, msg) => { console.log((c ? "PASS " : "FAIL ") + msg); if (!c) process.exitCode = 1; };
const base = "http://127.0.0.1:" + (process.env.WEB_PORT || 5173) + "/";
const box = p.getByRole("searchbox");
const results = () => p.locator("article.result");
const settle = async () => { await p.waitForTimeout(800); await p.waitForFunction(() => !/Loading/.test(document.querySelector(".result-count")?.innerText || "")); };
const firstTitle = async () => (await results().first().locator(".rtitle").innerText());
await p.goto(base); await p.waitForSelector("article.result");

// browse mode
ok((await results().count()) === 600, "empty search lists all 600 questions");
ok((await p.locator("h2.chapter-head").count()) === 36, "questions are grouped under 36 chapter headings");
// "/" focuses search
await p.locator("body").click({ position: { x: 5, y: 5 } }); await p.keyboard.press("/");
ok(await box.evaluate((el) => el === document.activeElement), "/ focuses the search box");

// id, typo, acronym, meaning
await box.fill("316"); await settle();
ok((await firstTitle()).includes("Multi-Head Attention") && (await results().first().innerText()).includes("ID match"), "an ID finds that question first, labelled ID match");
await box.fill("attenion"); await settle();
ok((await p.locator(".didyoumean").innerText()).includes("attention"), "a typo is corrected and said so");
ok((await results().count()) > 5, "corrected search returns results");
await box.fill("kl divergence"); await settle();
ok((await results().first().locator("mark").count()) > 0, "matching words are marked in the title");
ok((await results().first().innerText()).includes("KL"), "abbreviation search finds KL questions");
await box.fill("make training use less memory"); await settle();
const top5 = await results().evaluateAll((els) => els.slice(0, 5).map((e) => e.innerText));
ok(top5.some((t) => /Checkpointing|autocast|Accumulation/i.test(t)), "a description with none of the title words finds memory-saving questions");
ok((await results().first().innerText()).includes("Meaning"), "meaning matches are labelled");
ok((await p.locator(".result-count").innerText()).includes("keywords + meaning"), "status says keywords + meaning");

// syntax
await box.fill("attention -dropout"); await settle();
ok(!(await results().evaluateAll((els) => els.map((e) => e.innerText))).some((t) => /Apply Dropout to Attention Weights/.test(t)), "-dropout leaves that question out");
await box.fill("id:310-312"); await settle();
ok((await results().count()) === 3 && (await p.locator(".chips").innerText()).includes("id: 310-312"), "id range filter with a visible chip");
await box.fill("has:lesson"); await settle();
ok((await p.locator(".result-count").innerText()).startsWith("72 results"), "has:lesson finds the 72 selected lessons (planned or complete)");
await box.fill("lang:cpp"); await settle();
ok((await results().count()) === 9, "lang:cpp finds the 9 C++ questions");
await box.fill('"multi head"'); await settle();
ok((await results().first().innerText()).includes("Multi-Head"), "quoted phrase works");
await box.fill("zzzzqqqq"); await settle();
ok((await p.locator("main").innerText()).includes("No questions match") || (await results().count()) < 600, "nonsense gives a clear result");

// filters, sort, show more, clear
await box.fill("softmax"); await settle();
await p.getByRole("button", { name: "Question order" }).click();
const ids = await results().evaluateAll((els) => els.map((e) => Number(e.querySelector(".rid").innerText)));
ok(ids.length > 1 && ids.every((v, i) => i === 0 || v >= ids[i - 1]), "Question order sorts by ID");
await p.getByRole("button", { name: "Best match" }).click();
await box.fill("model"); await settle();
ok((await p.getByRole("button", { name: /Show 40 more/ }).count()) === 1, "long result lists offer Show more");
await p.getByRole("button", { name: /Show 40 more/ }).click(); await settle();
ok((await results().count()) === 67, "Show more adds the remaining results (model has 67)");
await p.getByRole("button", { name: "Clear" }).click(); await settle();
ok((await results().count()) === 600 && (await box.inputValue()) === "", "Clear restores the full list");
await p.getByLabel("Chapter").selectOption("C16"); await settle();
ok((await results().count()) === 20 && (await p.locator("h2.chapter-head").count()) === 1, "chapter menu filters the list to the 20 attention questions");
await p.getByRole("button", { name: "Clear" }).click();

// tips and keyboard
await p.locator("summary", { hasText: "Search tips" }).click();
await p.getByRole("button", { name: "chapter:attention softmax" }).click(); await settle();
ok((await box.inputValue()) === "chapter:attention softmax" && (await results().count()) > 0, "clicking an example runs it");
await box.focus(); await p.keyboard.press("ArrowDown");
ok(await p.evaluate(() => document.activeElement.classList.contains("result-link")), "ArrowDown moves into the results");
await p.keyboard.press("ArrowDown"); await p.keyboard.press("ArrowUp"); await p.keyboard.press("ArrowUp");
ok(await box.evaluate((el) => el === document.activeElement), "ArrowUp from the first result returns to the box");
await p.keyboard.press("Escape"); await settle();
ok((await box.inputValue()) === "", "Escape clears the search");

// weird input never breaks the page
for (const q of ['"', "((", "-", "a:b:c", "' OR 1=1 --", "<script>alert(1)</script>"]) {
  await box.fill(q); await settle();
  ok(!(await p.locator("[role=alert]").count()), `odd query ${JSON.stringify(q)} shows no error`);
}
await box.fill("");

// code viewer on a question page
await p.goto(base + "#/question/316");
await p.getByRole("button", { name: "Show solution" }).click();
await p.waitForSelector(".cm-content");
ok((await p.locator(".cm-gutters .cm-gutterElement").count()) > 10, "solution has line numbers");
ok((await p.locator(".cm-content .tok-keyword").count()) > 3, "keywords are syntax highlighted");
ok((await p.locator(".cm-content .tok-function, .cm-content .tok-def").count()) > 0, "function names are coloured");
const kw = (scheme) => p.evaluate(() => getComputedStyle(document.querySelector(".tok-keyword")).color);
const light = await kw(); await p.emulateMedia({ colorScheme: "dark" }); await p.waitForTimeout(150);
const dark = await kw();
ok(light !== dark, `colours change with the dark theme (${light} -> ${dark})`);
const bg = await p.evaluate(() => getComputedStyle(document.querySelector(".cm-editor")).backgroundColor);
ok(bg !== "rgb(246, 248, 250)", "code background follows the dark theme");
await p.emulateMedia({ colorScheme: "light" });
await p.getByRole("button", { name: "Copy" }).click();
ok((await p.getByRole("button", { name: "Copied" }).count()) === 1, "Copy confirms with the word Copied");
const clip = await p.evaluate(() => navigator.clipboard.readText());
const api = await (await fetch(base + "api/exercises/316/solution")).json();
ok(clip === api.code, "clipboard holds the exact solution code");
const wrapBefore = await p.locator(".cm-content.cm-lineWrapping").count();
await p.getByRole("button", { name: "Wrap lines" }).click();
ok((await p.locator(".cm-content.cm-lineWrapping").count()) === 1 && wrapBefore === 0, "Wrap lines toggles wrapping");
ok((await p.locator(".codebar").innerText()).includes("Python"), "language and line count shown");
// C++ question
await p.goto(base + "#/question/433"); await p.getByRole("button", { name: "Show solution" }).click(); await p.waitForSelector(".cm-content");
ok((await p.locator(".codebar").innerText()).includes("C++") && (await p.locator(".cm-content .tok-keyword").count()) > 0, "C++ solution is highlighted too");

// mobile
await p.setViewportSize({ width: 390, height: 800 }); await p.goto(base); await p.waitForSelector("article.result");
ok(!(await p.evaluate(() => document.documentElement.scrollWidth > innerWidth)), "search page has no sideways scroll at 390px");
await p.goto(base + "#/question/316"); await p.getByRole("button", { name: "Show solution" }).click(); await p.waitForSelector(".cm-content");
ok(!(await p.evaluate(() => document.documentElement.scrollWidth > innerWidth)), "question page with code has no sideways scroll at 390px");
await p.setViewportSize({ width: 1100, height: 900 });
await p.screenshot({ path: process.argv[2] + "/code.png" });
console.log("console errors:", errs.length ? errs : "none");
await b.close();
